import torch.utils.data


def customDataloader(args):
    print("=> fetching img pairs in %s" % (args.data_dir))
    if args.dataset == 'PS_Synth_Dataset':
        from datasets.PS_Synth_Dataset import PS_Synth_Dataset
        train_set = PS_Synth_Dataset(args, args.data_dir, 'train')
        val_set = PS_Synth_Dataset(args, args.data_dir, 'val')
    else:
        raise Exception('Unknown dataset: %s' % (args.dataset))

    if args.concat_data:
        print('****** Using cocnat data ******')
        print("=> fetching img pairs in %s" % (args.data_dir2))
        train_set2 = PS_Synth_Dataset(args, args.data_dir2, 'train')
        val_set2 = PS_Synth_Dataset(args, args.data_dir2, 'val')
        train_set = torch.utils.data.ConcatDataset([train_set, train_set2])
        val_set = torch.utils.data.ConcatDataset([val_set, val_set2])

    print('\t Found Data: %d Train and %d Val' % (len(train_set), len(val_set)))
    print('\t Train Batch %d, Val Batch: %d' % (args.batch, args.val_batch))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch,
                                               num_workers=args.workers, pin_memory=args.cuda, shuffle=True,
                                               collate_fn=custom_collate_fn_PSFCN, prefetch_factor=6,
                                               persistent_workers=False)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=args.val_batch,
                                              num_workers=args.workers, pin_memory=args.cuda, shuffle=False,
                                              collate_fn=custom_collate_fn_PSFCN, prefetch_factor=6,
                                              persistent_workers=False)
    return train_loader, test_loader


def benchmarkLoader(args):
    print("=> fetching img pairs in data/%s" % (args.benchmark))
    if args.benchmark == 'DiLiGenT_main':
        from datasets.DiLiGenT_main import DiLiGenT_main
        test_set = DiLiGenT_main(args, 'test')
    elif args.benchmark == "DiLiGenT_102_main":
        from datasets.DiLiGenT_102_main import DiLiGenT_102_main
        test_set = DiLiGenT_102_main(args, 'test')
    elif args.benchmark == "DiLiGenT_Pi_main":
        from datasets.DiLiGenT_Pi_main import DiLiGenT_Pi_main
        test_set = DiLiGenT_Pi_main(args, 'test')
    elif args.benchmark == 'Sphere_Bunny_main':
        from datasets.Sphere_Bunny_main import SphereBunny_main
        test_set = SphereBunny_main(args, 'test')

    else:
        raise Exception('Unknown benchmark')

    print('\t Found Benchmark Data: %d samples' % (len(test_set)))
    print('\t Test Batch %d' % (args.test_batch))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch,
                                              num_workers=args.test_batch, pin_memory=args.cuda, shuffle=False,
                                              prefetch_factor=3)
    return test_loader


'''以下2个合并的方法，仅针对Blobby/Sculpture数据集'''


def custom_collate_fn_PSFCN(batch):
    # 初始化一个字典来存储批量数据
    Normals, Imgs, Masks, Lights = zip(*batch)

    # 将列表转换为张量，并在第一个维度上进行拼接
    Loader_Data = {'N': torch.stack(Normals, dim=0), 'img': torch.stack(Imgs, dim=0), 'mask': torch.stack(Masks, dim=0),
                   'light': torch.stack(Lights, dim=0)}
    del Normals, Imgs, Masks, Lights, batch

    return Loader_Data


def custom_collate_fn_UPSFCN(batch):
    # 初始化一个字典来存储批量数据
    Normals, Imgs, Masks = zip(*batch)

    # 将列表转换为张量，并在第一个维度上进行拼接
    Loader_Data = {'N': torch.stack(Normals, dim=0), 'img': torch.stack(Imgs, dim=0), 'mask': torch.stack(Masks, dim=0)}

    del Normals, Imgs, Masks, batch

    return Loader_Data
