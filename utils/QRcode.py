import os.path
import random

import qrcode
import torchvision.transforms as T
import torchvision.transforms.functional as F
from pyzxing import BarCodeReader
from torchvision.io import read_image, ImageReadMode


class QRcode:
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=1,
            border=1,
        )
        self.reader = BarCodeReader()

    def message2img(self, message):
        file_name = str(random.randint(1, 10000)) + '.png'
        self.qr.clear()
        self.qr.add_data(message)
        self.qr.make(fit=True)
        self.qr.make_image(fill_color="black", back_color="white").save(os.path.join(self.cache_path, file_name))
        ret = T.Resize(size=(96, 96))(read_image(os.path.join(self.cache_path, file_name), ImageReadMode.GRAY))
        os.remove(os.path.join(self.cache_path, file_name))
        return ret

    def img2message(self, img):
        file_name = str(random.randint(1, 10000)) + '.png'
        F.to_pil_image(img).save(os.path.join(self.cache_path, file_name))
        results = self.reader.decode(os.path.join(self.cache_path, file_name))
        os.remove(os.path.join(self.cache_path, file_name))
        print(results[0]['parsed'])


if __name__ == "__main__":
    # x = QRcode().message2img('111')
    # QRcode().img2message(x)
    message = ''.join(str(i) for i in random.choices([0, 1], k=96))
    x = QRcode('../.cache').message2img(message)
    QRcode('../.cache').img2message(x)
