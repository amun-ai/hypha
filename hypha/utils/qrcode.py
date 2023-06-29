# Code adopted from https://github.com/Khalil-Youssefi/qrcodeT/tree/master
# Released under MIT license

import qrcode
import numpy as np


def qrcode2text(img):
    bindata = np.array(img)[::10, ::10] + 0
    if bindata.shape[0] % 2 != 0:
        bindata.resize((bindata.shape[0] + 1, bindata.shape[1]), refcheck=False)
    twolines_compress = bindata[::2, :] + 2 * bindata[1::2, :]
    chars = np.zeros(twolines_compress.shape)
    char_table = [" ", "▀", "▄", "█"]
    chars[twolines_compress == 0] = ord(char_table[0])
    chars[twolines_compress == 1] = ord(char_table[1])
    chars[twolines_compress == 2] = ord(char_table[2])
    chars[twolines_compress == 3] = ord(char_table[3])
    return chars


def generate_qrcode(txt, fill_color="black", back_color="white"):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=2,
    )
    qr.add_data(txt)
    qr.make(fit=True)
    img = qr.make_image(fill_color=fill_color, back_color=back_color)
    return img


def print_qrcode(txt):
    chars = qrcode2text(generate_qrcode(txt))
    for i in range(chars.shape[0]):
        for j in range(chars.shape[1]):
            print(chr(int(chars[i, j])), end="")
        print()


def qrcode2html(txt):
    chars = qrcode2text(generate_qrcode(txt))
    qrcode_str = ""
    for i in range(chars.shape[0]):
        for j in range(chars.shape[1]):
            qrcode_str += chr(int(chars[i, j]))
        qrcode_str += "\n"
    return f'<pre style="color:black!important; background: white;line-height: 14px; font-size=10px">{qrcode_str}</pre>'


def in_ipynb():
    try:
        cfg = get_ipython().config
        if isinstance(cfg, dict):
            return True
        else:
            return False
    except NameError:
        return False


def display_qrcode(text):
    if in_ipynb():
        from IPython.display import display, HTML

        qrcode_text = qrcode2html(text)
        display(
            HTML(
                f'<pre style="color: white;background: black;line-height: 14px; font-size=10px">{qrcode_text}</pre>'
            )
        )
    else:
        print_qrcode(text)


if __name__ == "__main__":
    display_qrcode("https://ai.imjoy.io")
