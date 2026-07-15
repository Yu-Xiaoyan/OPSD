"""OPSDv2Trainer — 三路分诊 + 错支目标手术（v2 主案）。

预注册见 docs/v2_launch_prereg.md。基底 = repo 冻结口径（clip 0.05、TM-off、solution、gb30）。
每 rollout 经 verifier-v2 分桶（correct / wrong / truncated；truncated→按 wrong 处理）。
teacher=base 固定。per-rollout 前向：S(grad) + π_T(base) + [π_T̃(base, 仅wrong+C′)] + [π_S0(base student, 仅B)]。

目标 logits 重构（renorm 后 forward-KL(target‖S)）：
  log_tgt = log π_T
  + λ·max(δ,0)     （C′-1，仅 wrong 支；δ = log π_T − log π_T̃）
  − γ·log π_S0     （B 全局漂移扣除，所有支，若开）
correct 支 loss × ω；wrong 支 × 1。流式：单样本 [T,V]，用完即 del。
"""
from __future__ import annotations
import os, sys
import torch
import torch.nn.functional as F
from opsd_trainer import OPSDTrainer

_PROBES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probes")
if _PROBES not in sys.path:
    sys.path.insert(0, _PROBES)
from scoring import (build_teacher_prompt_text, build_student_prompt_text,  # noqa: E402
                     forward_rollout_logits, teacher_mode)
from corrupt_solution import corrupt_solution  # noqa: E402
from corrupt_answers import corrupt_answer  # noqa: E402
from verify_answer import bucket_rollout_v2  # noqa: E402


class OPSDv2Trainer(OPSDTrainer):
    def __init__(self, *args, v2_use_B=True, v2_use_cp=True, v2_omega=1.0,
                 v2_lambda=1.0, v2_gamma=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.v2_use_B = bool(v2_use_B)
        self.v2_use_cp = bool(v2_use_cp)
        self.v2_omega = float(v2_omega)
        self.v2_lambda = float(v2_lambda)
        self.v2_gamma = float(v2_gamma)
        print(f"[OPSDv2Trainer] B={self.v2_use_B} C'={self.v2_use_cp} "
              f"omega={self.v2_omega} lambda={self.v2_lambda} gamma={self.v2_gamma}")

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if self._signature_columns is not None:
            for c in ("Answer", "solution"):
                if c not in self._signature_columns:
                    self._signature_columns.append(c)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.use_thinking_machines_loss:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        tok = self.processing_class
        sp_len = inputs["student_prompt_length"]
        sampled = inputs["student_input_ids"][:, sp_len:]
        shifted = inputs["labels"][:, sp_len:]
        mask = shifted != -100
        problems = inputs.get("problems"); gts = inputs.get("gt_answers"); sols = inputs.get("solutions")
        dev = inputs["student_input_ids"].device

        # 单次批量 student forward（唯一梯度图；DeepSpeed 只规约一次）
        out_s = model(input_ids=inputs["student_input_ids"],
                      attention_mask=inputs["student_attention_mask"])
        student_logits = out_s.logits[:, sp_len - 1:-1, :]

        B = sampled.shape[0]
        total = torch.zeros((), device=dev); wsum = 0.0
        n_bucket = {"correct": 0, "wrong": 0, "truncated": 0}
        for i in range(B):
            m_i = mask[i]
            if int(m_i.sum()) < 4:
                continue
            ids = sampled[i][m_i].tolist()
            S = student_logits[i][m_i].float()               # [T,V] 批量图切片
            problem = problems[i] if problems else ""
            gt = str(gts[i]) if gts else ""
            sol = sols[i] if sols else ""
            if not sol or not gt:
                continue
            # 分桶（verifier-v2；truncated→按 wrong）
            bucket, _ = bucket_rollout_v2(tok.decode(ids), gt, problem)
            n_bucket[bucket] = n_bucket.get(bucket, 0) + 1
            is_correct = (bucket == "correct")
            wrong_branch = not is_correct                     # wrong + truncated
            with torch.no_grad(), teacher_mode(model):
                TS, _, _ = forward_rollout_logits(model, tok, build_teacher_prompt_text(tok, problem, sol), ids)
                TSt = None
                if wrong_branch and self.v2_use_cp:
                    corr = corrupt_answer(gt, n=1, seed=i)
                    if corr:
                        csol, nrep = corrupt_solution(sol, gt, corr[0])
                        if nrep > 0:
                            TSt, _, _ = forward_rollout_logits(model, tok, build_teacher_prompt_text(tok, problem, csol), ids)
                S0 = None
                if self.v2_use_B:
                    S0, _, _ = forward_rollout_logits(model, tok, build_student_prompt_text(tok, problem), ids)
            T = S.shape[0]
            for X in (TS, TSt, S0):
                if X is not None:
                    T = min(T, X.shape[0])
            S = S[:T]; logTS = F.log_softmax(TS[:T].to(dev), dim=-1)
            log_tgt = logTS.clone()
            if TSt is not None:                               # C′-1
                delta = logTS - F.log_softmax(TSt[:T].to(dev), dim=-1)
                log_tgt = log_tgt + self.v2_lambda * delta.clamp_min(0)
                del delta
            if S0 is not None:                                # B 漂移扣除
                log_tgt = log_tgt - self.v2_gamma * F.log_softmax(S0[:T].to(dev), dim=-1)
            target = F.log_softmax(log_tgt, dim=-1)
            logS = F.log_softmax(S, dim=-1)
            elem = target.exp() * (target - logS)             # forward-KL 逐元素贡献 [T,V]
            if self.jsd_token_clip is not None:               # 保留 clip 0.05（冻结口径，P-a 净正）
                elem = elem.clamp(max=self.jsd_token_clip)
            kl = elem.sum(-1)                                  # [T]
            loss_i = kl.mean()
            del elem
            w = self.v2_omega if is_correct else 1.0
            total = total + w * loss_i * T; wsum += w * T
            del S, TS, logTS, log_tgt, target, logS, kl
            if TSt is not None: del TSt
            if S0 is not None: del S0

        loss = total / max(1e-8, wsum)
        mode = "train" if model.training else "eval"
        nb = sum(n_bucket.values()) or 1
        for k in ("correct", "wrong", "truncated"):
            self._metrics[mode][f"v2/frac_{k}"].append(n_bucket.get(k, 0) / nb)
        del out_s, student_logits
        torch.cuda.empty_cache()
        if return_outputs:
            class _O:
                def __init__(s, l): s.loss = l
            return (loss, _O(loss))
        return loss
