from dataloaders.datasets import apollo
from torch.utils.data import DataLoader

IMAGE_SIZE = [(1024, 384), (1536,512), (3384,1020), (2048, 768), (1636, 640)]
image_size = IMAGE_SIZE[0]

# 方式1:
def make_data_loader(args, **kwargs):
    if args.dataset == 'apollo':
        train_set = apollo.ApolloSegmentation(args, "train_lite.csv", mode='train')
        val_set = apollo.ApolloSegmentation(args, "val_lite.csv", mode='val')
        test_set = apollo.ApolloSegmentation(args, "test_lite.csv", mode='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True ,**kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=True, **kwargs)

        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError

# 方式2:
def make_data_loader2(args):
    num_class = 8
    train_loader = apollo.apollo_image_gen("train_lite.csv", mode='train', batch_size=args.batch_size, image_size=image_size, crop_offset=690)
    val_loader = apollo.apollo_image_gen("val_lite.csv", mode='val', batch_size=args.batch_size, image_size=image_size, crop_offset=690)
    test_loader = apollo.apollo_image_gen("test_lite.csv", mode='test' , batch_size=args.test_batch_size, image_size=image_size, crop_offset=690)
    return train_loader, val_loader, test_loader, num_class