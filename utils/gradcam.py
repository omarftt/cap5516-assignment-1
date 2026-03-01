import math
import numpy as np
import torch
import torch.nn.functional as F
import cv2


class ViTGradCAM:
    def __init__(self, model):
        self.model       = model
        self.activations = None
        self.gradients   = None

        target = model.get_last_block()
        self._fwd = target.register_forward_hook(self._save_act)
        self._bwd = target.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, input, output):
        self.activations = output.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int = None):
        self.model.eval()
        x = x.requires_grad_(True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        act  = self.activations[0, 1:, :]
        grad = self.gradients[0, 1:, :]

        weights = grad.mean(dim=0)
        cam = (act * weights).sum(dim=-1)
        cam = F.relu(cam)

        n = cam.shape[0]
        h = w = int(math.sqrt(n))
        cam = cam.reshape(h, w).cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        return cam

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()


def overlay_cam(image_np: np.ndarray, cam: np.ndarray):
    """Overlay images for visualization"""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (0.5 * image_np + 0.5 * heatmap).astype(np.uint8)