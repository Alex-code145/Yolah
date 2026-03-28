import torch
import torch.nn.functional as F
from .nn_model import state_to_tensor, OUTPUT_POLICY_SIZE
from Jeu.Yolah import Move

# evaluator wraps a PyTorch model and returns policy probs over legal moves and a scalar value.
# It expects the model to return (policy_logits, value_tensors) when fed a tensor input.

class NeuralEvaluator:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def __call__(self, state):
        # state: YolahState
        # convert to tensor
        black, white, empty, black_score, white_score, ply = state.game.get_state()
        current_player = state.current_player()
        inp = state_to_tensor(black, white, empty, current_player)
        inp = inp.unsqueeze(0).to(self.device)  # (1,3,8,8)

        with torch.no_grad():
            logits, value = self.model(inp)
            # logits: (1, 4096), value: (1,)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            value = float(value.cpu().numpy()[0])

        # map probs to legal moves only
        legal = state.legal_moves()
        move_probs = {}
        total = 0.0
        for m in legal:
            idx = self.move_to_index(m)
            p = float(probs[idx])
            move_probs[str(m)] = p
            total += p

        # renormalize
        if total > 0:
            for k in move_probs:
                move_probs[k] /= total
        else:
            # fallback: uniform over legal moves
            unif = 1.0 / len(legal)
            for m in legal:
                move_probs[str(m)] = unif

        return move_probs, value

    def batch_eval(self, states):
        """Evaluate a list of YolahState objects in a single batched model call.

        Returns a list of (move_probs, value) tuples in the same order as states.
        """
        # build batch tensor
        tensors = []
        for state in states:
            black, white, empty, black_score, white_score, ply = state.game.get_state()
            current_player = state.current_player()
            inp = state_to_tensor(black, white, empty, current_player)
            tensors.append(inp)
        batch = torch.stack(tensors, dim=0).to(self.device)

        with torch.no_grad():
            logits, values = self.model(batch)
            probs_batch = F.softmax(logits, dim=1).cpu().numpy()
            values = values.cpu().numpy().reshape(-1)

        results = []
        for i, state in enumerate(states):
            probs = probs_batch[i]
            legal = state.legal_moves()
            move_probs = {}
            total = 0.0
            for m in legal:
                idx = self.move_to_index(m)
                p = float(probs[idx])
                move_probs[str(m)] = p
                total += p

            if total > 0:
                for k in move_probs:
                    move_probs[k] /= total
            else:
                unif = 1.0 / len(legal)
                for m in legal:
                    move_probs[str(m)] = unif

            results.append((move_probs, float(values[i])))

        return results

    @staticmethod
    def move_to_index(move: Move):
        return move.from_sq.sq * 64 + move.to_sq.sq
