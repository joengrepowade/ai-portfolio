import torch
import torchvision.transforms as T
import numpy as np
from typing import List, Tuple
import cv2


class VideoFrameSampler:
    def __init__(self, n_frames=8, sample_mode='uniform', fps=None):
        self.n_frames = n_frames
        self.sample_mode = sample_mode
        self.fps = fps

    def sample(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self._get_indices(total)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()
        return np.stack(frames) if frames else np.zeros((self.n_frames, 224, 224, 3), dtype=np.uint8)

    def _get_indices(self, total: int) -> List[int]:
        if self.sample_mode == 'uniform':
            return np.linspace(0, total - 1, self.n_frames, dtype=int).tolist()
        elif self.sample_mode == 'random':
            return sorted(np.random.choice(total, self.n_frames, replace=total < self.n_frames).tolist())
        elif self.sample_mode == 'dense':
            start = np.random.randint(0, max(1, total - self.n_frames))
            return list(range(start, start + self.n_frames))
        return list(range(min(self.n_frames, total)))


class VideoTransform:
    def __init__(self, img_size=224, mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225), train=True):
        transforms = []
        if train:
            transforms += [
                T.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ]
        else:
            transforms += [T.Resize(int(img_size * 1.15)), T.CenterCrop(img_size)]
        transforms += [T.ToTensor(), T.Normalize(mean, std)]
        self.transform = T.Compose(transforms)

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        # frames: (T, H, W, C)
        from PIL import Image
        tensors = [self.transform(Image.fromarray(f)) for f in frames]
        video = torch.stack(tensors)          # (T, C, H, W)
        return video.permute(1, 0, 2, 3)      # (C, T, H, W)


def collate_video_batch(samples: List[Tuple]) -> dict:
    videos = torch.stack([s[0] for s in samples])
    labels = torch.tensor([s[1] for s in samples], dtype=torch.long)
    return {'video': videos, 'label': labels}
