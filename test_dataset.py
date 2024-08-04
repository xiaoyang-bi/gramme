from datasets.sequence_folders_mono import SequenceFolder

if __name__ == '__main__':
    dataset = SequenceFolder(root = '../data/radiate', dataset='radiate', seed=3407, preprocessed=True)
    tgt_img, ref_imgs, intrinsics = dataset[0]
    print(len(dataset))
    print(tgt_img)
    print(len(ref_imgs))
    print(intrinsics)