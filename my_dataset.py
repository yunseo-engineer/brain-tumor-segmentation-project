import torch
import numpy as np
import random
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from pathlib import Path

class BraTS25DDataset(Dataset):
    def __init__(self, proc_dir, split, n_adj=2, oversample=0.7, augment=False, cache_size=40):
        self.split      = split
        self.proc_dir   = proc_dir / split
        self.n_adj      = n_adj
        self.oversample = oversample
        self.augment    = augment
        self.cache_size = cache_size
        self._cache     = {}          # pid → {'image':..., 'mask':..., 'valid_axial':...}
        self._cache_order = []        # LRU 순서

        self.tumor_samples = []       # (pid, z)
        self.other_samples = []

        # 인덱스 구축
        npz_files = sorted(self.proc_dir.glob('*.npz'))
        for npz_path in tqdm(npz_files, desc=f'[{split}] 인덱스 구축', leave=False):
            pid   = npz_path.stem
            data  = self._load(pid, npz_path)
            mask  = data['mask']      # (192,192,D,3)
            for z in data['valid_axial'].tolist():
                has_t = bool(mask[:,:,z,:].sum() > 0)
                if has_t:
                    self.tumor_samples.append((pid, z))
                else:
                    self.other_samples.append((pid, z))

        self.all_samples = self.tumor_samples + self.other_samples
        nt, no = len(self.tumor_samples), len(self.other_samples)
        print(f'[{split}] 총 {len(self.all_samples)} slices '
              f'(종양:{nt} ({100*nt/max(1,nt+no):.1f}%), 비종양:{no})')

    def _load(self, pid, path=None):
        if pid not in self._cache:
            if len(self._cache) >= self.cache_size:
                evict = self._cache_order.pop(0)
                del self._cache[evict]
            if path is None:
                path = self.proc_dir / f'{pid}.npz'
            
            raw = np.load(path)
            self._cache[pid] = {
                'image':       raw['image'].astype(np.float32),   # (192,192,D,4)
                'mask':        raw['mask'].astype(np.float32),    # (192,192,D,3)
                'valid_axial': raw['valid_axial']
            }
            self._cache_order.append(pid)
        else:
            self._cache_order.remove(pid)
            self._cache_order.append(pid)
        return self._cache[pid]

    def __len__(self): 
        return len(self.all_samples)

    def _get_stack(self, image, z_center):
        D       = image.shape[2]
        z_idx   = np.clip(np.arange(z_center - self.n_adj,
                                    z_center + self.n_adj + 1), 0, D - 1)
        return np.concatenate([image[:,:,zi,:] for zi in z_idx], axis=-1)

    def _augment(self, img, mask):
        if random.random() < 0.5:
            img = img[:, ::-1, :].copy(); mask = mask[:, ::-1, :].copy()
        if random.random() < 0.5:
            img = img[::-1, :, :].copy(); mask = mask[::-1, :, :].copy()
        if random.random() < 0.3:
            k = random.choice([1,2,3])
            img  = np.rot90(img,  k, (0,1)).copy()
            mask = np.rot90(mask, k, (0,1)).copy()
        if random.random() < 0.3:
            img = img + np.random.normal(0, 0.05, img.shape).astype(np.float32)
        return img, mask

    def __getitem__(self, idx):
        if self.split == 'train' and self.tumor_samples and random.random() < self.oversample:
            pid, z = random.choice(self.tumor_samples)
        elif self.split == 'train' and self.other_samples:
            pid, z = random.choice(self.other_samples)
        else:
            pid, z = self.all_samples[idx]

        data   = self._load(pid)
        img_5  = self._get_stack(data['image'], z)    # (192,192,20)
        mask_z = data['mask'][:,:,z,:].copy()          # (192,192,3)

        if self.augment:
            img_5, mask_z = self._augment(img_5, mask_z)

        img_t  = torch.from_numpy(img_5).permute(2,0,1).float()
        mask_t = torch.from_numpy(mask_z).permute(2,0,1).float()
        return img_t, mask_t