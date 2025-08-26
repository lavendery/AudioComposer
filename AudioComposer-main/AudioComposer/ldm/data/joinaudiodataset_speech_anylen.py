# AudioComposer Team, 2024. All Rights Reserved. 
# https://arxiv.org/abs/2409.12560

import sys
import numpy as np
import torch
import logging
from ldm.data.joinaudiodataset_anylen import *
logger = logging.getLogger(f'main.{__name__}')
import torchaudio
sys.path.insert(0, '.')
from datasets import load_dataset


class JoinManifestSpecs(torch.utils.data.Dataset):
    def __init__(self, split, train_path, validation_path, text_column, audio_column, min_factor=1,  min_batch_len=8000, mode='pad', spec_crop_len=1248, pad_value=-5):
        super().__init__()

        self.max_batch_len = spec_crop_len
        self.min_batch_len = min_batch_len
        self.min_factor = min_factor
        self.pad_value = pad_value
        assert mode in ['pad','tile']
        self.collate_mode = mode

        # split : train or valid 
        extension = train_path.split(".")[-1]
        if split == 'train':
            self.dataset = load_dataset(extension, data_files={'train': train_path})['train']
        elif split == 'valid':
            self.dataset = load_dataset(extension, data_files={'validation': validation_path})['validation']
        else:
            raise ValueError(f'Unknown split {split}')
        self.indices = list(range(len(self.dataset)))

        self.inputs = []
        self.audios = []
        self.pitch_category = []
        self.energy_category = []
        for index in self.indices:
            if self.dataset[index]["dataset"] != "strong" and self.dataset[index]["dataset"] != "audiocaps": 
                self.inputs.append(self.dataset[index][text_column])
                self.audios.append(self.dataset[index][audio_column])
                self.pitch_category.append(self.dataset[index]["pitch_category"])
                self.energy_category.append(self.dataset[index]["energy_category"])

    def __len__(self):
        return len(self.dataset)

    def get_num_instances(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = {}
        if self.dataset[index]["dataset"] == "strong" or self.dataset[index]["dataset"] == "audiocaps":
            text = self.dataset[index]['data_numpy'][0]
            start_times = [round(float(time) * 100) for time in self.dataset[index]['data_numpy'][1]]
            end_times = [round(float(time) * 100) for time in self.dataset[index]['data_numpy'][2]]
            # pitch_category = False
            # energy_category = False
            cur_audio = self.dataset[index]['location']
            waveform, old_sample_rate = torchaudio.load(cur_audio)
            waveform = waveform[0]
            sample_rate = 16000
            if old_sample_rate != sample_rate:
                waveform = torchaudio.functional.resample(waveform, orig_freq=old_sample_rate, new_freq=sample_rate)
            
            # return waveform, start_times, end_times, text, pitch_category, energy_category
            combine_caption = ''
            for i in range(len(text)):
                # combine_caption += text[i] + ', Start Time: ' + str(start_times[i]) + ', End Time: ' + str(end_times[i]) + '. '
                combine_caption += text[i] + ', Start at {:.2f}s and End at {:.2f}s. '.format(start_times[i]/100, end_times[i]/100)
            item['audio'] = waveform
            item["caption"] = combine_caption
        else:
            # randomly select number of mixes between 2 and 10
            num_mixes = np.random.randint(2, 11)
            # randomly select num_mixes audios
            idxs = np.random.choice(len(self.audios), num_mixes, replace=False)
            sample_rate = 16000
            mixed_waveform = torch.zeros(int(10 * sample_rate))
            start_times = []
            end_times = []

            bg_wav = False
            stop_mix = False
            fixed_time_min = 4
            fixed_time_max = 10
            text = []
            selected_file = []
            pitch_category = []
            energy_category = []
            for i in range(num_mixes):
                waveform, old_sample_rate = torchaudio.load(self.audios[idxs[i]])
                waveform = waveform[0]
                if old_sample_rate != sample_rate:
                    waveform = torchaudio.functional.resample(waveform, orig_freq=old_sample_rate, new_freq=sample_rate)

                # random select
                if len(waveform) < fixed_time_min * sample_rate:
                    waveform = waveform.repeat(fixed_time_min * sample_rate // len(waveform) + 1)[:fixed_time_min * sample_rate]
                
                min_length = 2.5 * 100  # min 2.5 seconds
                max_length = min(fixed_time_max * 100, len(waveform) // 160) # max 10 seconds
                cur_length = np.random.randint(min_length, max_length + 1) 

                start = 0  
                waveform = waveform[start:start+cur_length * 160]
                
                if len(start_times) == 0:
                    start_time = 0
                else:
                    ran = np.random.random()
                    if ran < 0.3:
                        # random blank 0-2s
                        start_time = end_times[-1] + int(np.random.random() * 2 * 100)
                    elif ran < 0.5:
                        # stop mix
                        start_time = 10 * 100
                    else:
                        start_time = end_times[-1]

                # if length of mixed_waveform exceeds 10 seconds, stop
                if start_time >= 10 * 100:
                    break
                
                if start_time+cur_length > 10 * 100:
                    cur_length = 10 * 100 - start_time
                    waveform = waveform[:cur_length * 160]
                    stop_mix = True
                
                # if length of current waveform is less than 1 second, stop the mixing
                if cur_length < 1 * 100:
                    break

                # mix
                mixed_waveform[(start_time * 160):(start_time+cur_length) * 160] += waveform
                selected_file.append(self.audios[idxs[i]])
                text.append(self.inputs[idxs[i]])
                pitch_category.append(self.pitch_category[idxs[i]])
                energy_category.append(self.energy_category[idxs[i]])

                # update time info
                start_times.append(start_time)
                end_times.append(start_time + cur_length)

                if stop_mix:
                    break

            # return mixed_waveform, start_times, end_times, text, selected_file
            # return mixed_waveform, start_times, end_times, text, pitch_category, energy_category
            combine_caption = ''
            for i in range(len(text)):
                # combine_caption += text[i] + ', Start Time: ' + str(start_times[i]) + ', End Time: ' + str(end_times[i]) + ', Audio Pitch: ' + str(pitch_category[i]) + ', Audio Energy: ' + str(energy_category[i]) + '. '
                combine_caption += text[i] + ', Start at {:.2f}s and End at {:.2f}s, it has {} and {}. '.format(start_times[i]/100, end_times[i]/100, pitch_category[i].title(), energy_category[i].title())
            item['audio'] = mixed_waveform
            item["caption"] = combine_caption
        return item

    def collate_fn(self, inputs):
        to_dict = {}
        for l in inputs:
            for k,v in l.items():
                if k in to_dict:
                    to_dict[k].append(v)
                else:
                    to_dict[k] = [v]

        if self.collate_mode == 'pad':
            to_dict['audio'] = collate_1d_or_2d(to_dict['audio'],pad_idx=self.pad_value, min_len = self.min_batch_len, max_len=self.max_batch_len, min_factor=self.min_factor)
        elif self.collate_mode == 'tile':
            to_dict['audio'] = collate_1d_or_2d_tile(to_dict['audio'],min_len = self.min_batch_len,max_len=self.max_batch_len,min_factor=self.min_factor)
        else:
            raise NotImplementedError
        
        return to_dict

class JoinSpecsTrain(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        print('specs_dataset_cfg ', specs_dataset_cfg)
        super().__init__('train', **specs_dataset_cfg)

class JoinSpecsValidation(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('valid', **specs_dataset_cfg)

class JoinSpecsTest(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('test', **specs_dataset_cfg)
