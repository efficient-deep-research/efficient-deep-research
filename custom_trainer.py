from typing import Tuple

import torch
import torch.nn.functional as F
from swift.trainers import DPOTrainer
from trl.trainer.utils import selective_log_softmax


class SearchResultMaskDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.begin_search_result_ids = [
            torch.tensor(self.tokenizer("<|begin_search_result|>")["input_ids"]),
            torch.tensor(self.tokenizer(" <|begin_search_result|>")["input_ids"]),
            torch.tensor(self.tokenizer("<|begin_search_result|>-")["input_ids"]),
            torch.tensor(self.tokenizer(" <|begin_search_result|>-")["input_ids"]),
        ]
        self.end_search_result_ids = [
            torch.tensor(self.tokenizer("<|end_search_result|>")["input_ids"]),
            torch.tensor(self.tokenizer(" <|end_search_result|>")["input_ids"]),
            torch.tensor(self.tokenizer("<|end_search_result|>\n\n")["input_ids"]),
            torch.tensor(self.tokenizer(" <|end_search_result|>\n\n")["input_ids"]),
            torch.tensor(self.tokenizer(".<|end_search_result|>\n\n")["input_ids"]),
            torch.tensor(self.tokenizer('."<|end_search_result|>\n\n')["input_ids"]),
            torch.tensor(self.tokenizer(")<|end_search_result|>\n\n")["input_ids"]),
        ]

    def get_per_token_logps(
        self, logits: torch.FloatTensor, labels: torch.LongTensor, label_pad_token_id=-100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                f"Logits (batch and sequence length dim) {logits.shape[:-1]}"
                "and labels must have the same shape {labels.shape}"
            )
        loss_mask = labels != label_pad_token_id

        # ================== Custom part =================
        # print("test|", repr(self.tokenizer.decode([26432, 408, 10716, 5287, 91, 1339])), "|end")
        try:
            loss_mask = loss_mask | self._mask_between_tags(labels)
        except Exception as e:
            print(f"💥Error occurred while masking between tags: {e}")
            for pattern in self.begin_search_result_ids:
                print("begin pattern:", pattern.tolist())
            for pattern in self.end_search_result_ids:
                print("end pattern:", pattern.tolist())
            for i in range(labels.shape[0]):
                print(f"Found search result tags in sequence {i}:")
                print(
                    "decoded labels:\n", self.tokenizer.decode(labels[i][labels[i] != -100], skip_special_tokens=False)
                )
                print("labels:\n", labels[i].tolist())
        # ================== Custom part =================
        
        labels = labels.clone()
        labels[~loss_mask] = 0
        if self.template.sequence_parallel_size == 1:
            # https://github.com/huggingface/trl/pull/2799
            # Reduce peak vram consumption with efficient selective log_softmax
            per_token_logps = selective_log_softmax(logits, labels)
            per_token_logps[~loss_mask] = 0
            return per_token_logps, logits.mean(-1), loss_mask
        else:
            labels = labels.to(logits.device)
            loss_mask = loss_mask.to(logits.device)
            mean_logits = logits.mean(-1)
            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
            from swift.trainers.sequence_parallel import sequence_parallel
            from swift.trainers.sequence_parallel.utils import GatherLoss

            position_ids = sequence_parallel.real_position_ids
            total_per_token_logps, total_loss_mask = GatherLoss.apply(per_token_logps, loss_mask, 1, position_ids)
            total_mean_logits = sequence_parallel.gather(mean_logits, dim=1, position_ids=position_ids)
            if position_ids is not None and position_ids.min() == -1:
                _pos_mask = position_ids >= 0
                total_per_token_logps = total_per_token_logps[_pos_mask].contiguous()
                total_mean_logits = total_mean_logits[_pos_mask].contiguous()
                total_loss_mask = total_loss_mask[_pos_mask].contiguous()

            total_loss_mask = total_loss_mask.bool()
            total_per_token_logps = total_per_token_logps * (total_loss_mask)

            if total_per_token_logps.dim() == 1:
                total_per_token_logps = total_per_token_logps.unsqueeze(0)
                total_mean_logits = total_mean_logits.unsqueeze(0)
                total_loss_mask = total_loss_mask.unsqueeze(0)
            return total_per_token_logps, total_mean_logits, total_loss_mask

    def _detect_tags(self, inputs: torch.Tensor, patterns: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Detect start positions of given subsequence patterns in a batched token sequence.

        Returns:
            match_flags: Bool tensor of shape (batch_size, seq_len).
            True at positions where any pattern starts; False elsewhere.
            match_lengths: Long tensor of shape (batch_size, seq_len).
            The length of the matching pattern at each start position; 0 where no pattern starts.
        """
        batch_size, seq_len = inputs.shape
        match_flags = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=inputs.device)
        match_lengths = torch.zeros((batch_size, seq_len), dtype=torch.long, device=inputs.device)

        for pattern in patterns:
            pattern = pattern.to(inputs.device)
            pattern_len = pattern.numel()
            if seq_len < pattern_len:
                continue

            match = (inputs.unfold(1, pattern_len, 1) == pattern.view(1, 1, pattern_len)).all(-1)
            match = F.pad(match, (0, pattern_len - 1))
            match_flags |= match
            match_lengths[match] = pattern_len

        return match_flags, match_lengths

    def _mask_between_tags(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = inputs.shape

        begin_flags, _ = self._detect_tags(inputs, self.begin_search_result_ids)
        end_flags, end_lengths = self._detect_tags(inputs, self.end_search_result_ids)

        mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=inputs.device)
        for batch_index in range(batch_size):
            start_positions = torch.nonzero(begin_flags[batch_index], as_tuple=False).squeeze(1).tolist()
            end_positions = torch.nonzero(end_flags[batch_index], as_tuple=False).squeeze(1).tolist()
            assert len(start_positions) == len(end_positions), "Mismatched number of begin and end tags"
            for start, end in zip(start_positions, end_positions):
                assert start < end, "Begin tag must come before end tag"
                end_tag_length = int(end_lengths[batch_index, end].item())  # この B の長さ
                end = min(end + end_tag_length - 1, seq_len - 1)  # B の末尾（含む）
                mask[batch_index, start : end + 1] = 1

        return mask.bool()
