from mxnet import gluon, image
import multiprocessing


class DataSet(gluon.data.Dataset):
    def __init__(self, root, DomainAList, DomainBList):
        self.root = root
        self.DomainAList = DomainAList
        self.DomainBList = DomainBList
        self.load_images()

    def read_images(self, root):
        Aroot = root + 'trainA/'  # left_frames   #data
        Broot = root + 'trainB/'  # labels   #label
        A, B = [None] * len(self.DomainAList), [None] * len(self.DomainBList)
        for i, name in enumerate(self.DomainAList):
            A[i] = image.imread(Aroot + name)
        for i,name in enumerate(self.DomainBList):
            B[i] = image.imread(Broot + name)

        return A, B

    def load_images(self):
        A, B = self.read_images(root=self.root)

        self.A = [self.normalize_image(im) for im in A]
        self.B = [self.normalize_image(im) for im in B]

        print('read ' + str(len(self.A)) + ' examples')
        print('read ' + str(len(self.B)) + ' examples')

    def normalize_image(self, A):
        return A.astype('float32') / 255

    def __getitem__(self, item):
        # A = image.imresize(self.A[item], 128, 128)
        # B = image.imresize(self.B[item], 128, 128)       #resize
        A = self.A[item]
        B = self.B[item]
        return A.transpose((2, 0, 1)), B.transpose((2, 0, 1))

    def __len__(self):
        return len(self.A)


def load_dataset(dir, A_list, B_list,batchsize):
    """"""
    dataset = DataSet(dir, A_list, B_list )
    data_iter = gluon.data.DataLoader(dataset, batchsize, shuffle=True, last_batch='discard', num_workers=multiprocessing.cpu_count()-1)

    return data_iter
