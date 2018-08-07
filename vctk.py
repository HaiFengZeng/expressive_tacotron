from nnmnkwii.datasets import vctk

vctk_dataset = vctk.WavFileDataSource('/home/zeng/work/data/VCTK-Corpus/')
print(vctk_dataset.collect_files())
