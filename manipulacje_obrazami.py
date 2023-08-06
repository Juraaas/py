from PIL import Image
import numpy as np
from PIL import ImageChops, ImageOps, ImageShow, ImageFilter
from PIL import ImageStat as stat
import matplotlib.pyplot as plt


def statystyki(im):
    s = stat.Stat(im)
    print("ekstrema ", s.extrema)
    print("zlicznik ", s.count)
    print("srednia ", s.mean)
    print("mediana ", s.median)
    print("odchylenie standartowe ", s.stddev)


im = Image.open('brain.png').convert('L')
print(im.mode)
statystyki(im)
print(im.size)

plt.title("obraz")
plt.axis('off')
plt.imshow(im, 'gray')
plt.show()

hist = im.histogram()
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot([i for i in range(0, 256)], hist)
plt.xticks(ticks=[], labels=[])
plt.title('Histogram')


def histogram_norm(img):
    return np.divide(img.histogram(), img.size[0] * img.size[1])


hist_normalizowany = histogram_norm(im)
plt.subplot(2, 2, 2)
plt.plot([i for i in range(0, 256)], hist_normalizowany)
plt.xticks(ticks=[], labels=[])
plt.title('Histogram normalizowany')


plt.subplot(2, 2, 3)


def histogram_cumul(obraz):
    hist_normalizowany = histogram_norm(obraz)
    hist_skumulowany = [hist_normalizowany[0]]
    for i in range(1, 256):
        hist_skumulowany.append(hist_skumulowany[i - 1] + hist_normalizowany[i])
    return hist_skumulowany


hist_skumulowany = histogram_cumul(im)
plt.plot([i for i in range(0, 256)], hist_skumulowany)
plt.xticks(ticks=[], labels=[])
plt.title('Histogram skumulowany')


def histogram_equalization(obraz):
    hist_skum = histogram_cumul(obraz)
    cp = obraz.copy()
    cp_l = cp.load()
    for i in range(0, obraz.size[0]):
        for j in range(0, obraz.size[1]):
            tmp = cp_l[i, j]
            cp_l[i, j] = int(255 * hist_skumulowany[tmp])
    return cp


wyr = histogram_equalization(im)
hist = wyr.histogram()
plt.subplot(2, 2, 4)
plt.bar([i for i in range(0, 256)], hist)
plt.xticks(ticks=[], labels=[])
plt.title('Histogram wyrównany')
plt.savefig('histogramy.jpg')
wyr.save('equalized.png')
wyr1 = ImageOps.equalize(im)
wyr1.save('equalized1.png')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.axis('off')
plt.title('equalized')
plt.imshow(wyr, 'gray')
plt.subplot(1, 2, 2)
plt.axis('off')
plt.title('equalized1')
plt.imshow(wyr1, 'gray')
plt.savefig('wyrownanie.jpg')
print('wyr', statystyki(wyr))
print('-----------------')
print('wyr1', statystyki(wyr1))


obraz = Image.open("wladek.png")
obraz1 = obraz.convert("RGBA")
r,g,b,a = obraz1.split()
obraz2 = obraz1.copy().convert("RGB")
obraz3 = Image.new("RGB", obraz2.size, (255, 255, 255))
obraz3.paste(obraz1)
obraz3.save("obraz3.png")
ImageChops.difference(obraz2, obraz3).show()


modes = ["CMYK", "YCbCr", "HSV"]
i = 1
plt.figure(figsize=(32, 16))
for m in modes:
    plt.subplot(1, 3, i)
    plt.title(str(m))
    plt.imshow(obraz3.convert(m))
    i += 1

plt.savefig("fig1.png")

cmyk = obraz3.copy().convert("CMYK")
ycbcr = obraz3.copy().convert("YCbCr")
hsv = obraz3.copy().convert("HSV")
c, m, y, k = cmyk.split()
Y, Cb, Cr = ycbcr.split()
h, s, v = hsv.split()
kanaly_cmyk = [c, m, y, k]
nazwy_cmyk = ["C", "M", "Y", "K"]
kanaly_ycbcr = [Y, Cb, Cr]
nazwy_ycbcr = ["Y", "Cb", "Cr"]
kanaly_hsv = [h, s, v]
nazwy_hsv = ["H", "S", "V"]

i = 1
j = 0
plt.figure(figsize=(32, 16))
for ch in kanaly_cmyk:
    plt.subplot(1, 4, i)
    plt.title(nazwy_cmyk[j])
    plt.imshow(ch, "gray")
    i += 1
    j += 1

plt.savefig("cmyk.png")

i = 1
j = 0
plt.figure(figsize=(32, 16))
for ch in kanaly_ycbcr:
    plt.subplot(1, 3, i)
    plt.title(nazwy_ycbcr[j])
    plt.imshow(ch, "gray")
    i += 1
    j += 1

plt.savefig("ycbcr.png")

i = 1
j = 0
plt.figure(figsize=(32, 16))
for ch in kanaly_hsv:
    plt.subplot(1, 3, i)
    plt.title(nazwy_hsv[j])
    plt.imshow(ch, "gray")
    i += 1
    j += 1

plt.savefig("hsv.png")

obraz4 = Image.open("wlodzimierz.jpg").resize(obraz1.size, 1)
obraz4_copy = obraz4.copy()
obraz1_copy = obraz1.copy()
obraz4_copy.paste(obraz1, mask=a)
obraz1_copy.paste(obraz4, mask=a)

plt.figure(figsize=(32, 16))
plt.subplot(1, 2, 1)
plt.title("Obraz 1 w obraz 4")
plt.imshow(obraz4_copy, "gray")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title("Obraz 4 w obraz 1")
plt.imshow(obraz1_copy, "gray")
plt.axis("off")
plt.savefig("fig2.png")



im = Image.open('goat.jpg')
print(im.size, im.mode)

w, h = im.size
s_w = 0.15
s_h = 0.27
s_wb = 1 / 0.15
s_hb = 1 / 0.27
im_N = im.resize((int(w*s_w), int(h*s_h)), 0)
im_Nb = im_N.resize((int(int(w*s_w)*s_wb), int(int(h*s_h)*s_hb)), 0)
im_r3 = im.resize((int(int(w * s_w) * s_wb), int(int(h * s_h) * s_hb)), 0)
resample_method = ['NEAREST', 'LANCZOS', 'BILINEAR', 'BICUBIC', 'BOX', 'HAMMING']
plt.figure(figsize=(20, 16))
i = 1
for t in range(6):
    file_name = "resample_" + str(resample_method[t])
    im_r = im.resize((int(w*s_w), int(h*s_h)), t)
    im_r2 = im_r.resize((int(int(w*s_w)*s_wb), int(int(h*s_h)*s_hb)), t)
    plt.subplot(6, 2, i)
    plt.title(str(file_name))
    #plt.imshow(im_r)
    #plt.axis('off')
    #i += 1
    #diff = ImageChops.difference(im_N, im_r)
    #s = stat.Stat(diff)
    #plt.subplot(6, 2, i)
    #plt.title('srednia' + str(np.round(s.mean, 2)))
    #plt.imshow(diff)
    #plt.axis('off')
    #i += 1

    #plt.subplot(6, 2, i)
    #plt.title(str(file_name))
    #plt.imshow(im_r2)
    #plt.axis('off')
    #i += 1
    #diff2 = ImageChops.difference(im_Nb, im_r2)
    #s = stat.Stat(diff2)
    #plt.subplot(6, 2, i)
    #plt.title('srednia' + str(np.round(s.mean, 2)))
    #plt.imshow(diff2)
    #plt.axis('off')
    #i += 1
#plt.savefig('test2.png')



print(im.size, im.mode)
print(im_Nb.size, im_Nb.mode)

w_p = 360
h_p = 80
w_k = 478
h_k = 250
wycinek = (w_p, h_p, w_k, h_k) # definicja miejsca wycięcia w_p, h_p - lewy górny róg, w_k,h_k prawy dolny róg
wyc_w = wycinek[2] - wycinek[0] # szerokość wycinka
wyc_h = wycinek[3] - wycinek[1] # wysokość wycinka
s_w = 2 # skala dla szerokości
s_h = 3 # skala dla wysokości
glowa = im.resize((s_w * wyc_w, s_h * wyc_h), box=wycinek)# wycina wycienek i zmienia rozmiar wycinka
#glowa.show()

glowa1 = im.crop(wycinek)
glowa1 = glowa1.resize((s_w * wyc_w, s_h * wyc_h))
#glowa1.show()
diff = ImageChops.difference(glowa, glowa1)

plt.figure(figsize=(32, 16))
plt.subplot(3, 1, 1)
plt.title("Glowa")
plt.imshow(glowa)
plt.axis("off")
plt.subplot(3, 1, 2)
plt.title("Glowa1")
plt.imshow(glowa1)
plt.axis("off")
plt.subplot(3, 1, 3)
plt.title("ImageChops.difference")
plt.imshow(diff)
plt.axis("off")
plt.savefig("fig_3.png")


obr60_1 = im.rotate(60, expand=1, fillcolor='red')
#obr60_1.show()
obr60_2 = im.rotate(60, expand=0, fillcolor='red')
#obr60_2.show()
obr60_3 = im.rotate(300, expand=1, fillcolor='green')
#obr60_3.show()
obr60_4 = im.rotate(300, expand=0, fillcolor='green')
#obr60_4.show()

plt.figure(figsize=(32, 16))
plt.subplot(2, 2, 1)
plt.title("Obrot 60 lewo, expand 1")
plt.imshow(obr60_1)
plt.axis("off")
plt.subplot(2, 2, 2)
plt.title("Obrot 60 prawo, expand 1")
plt.imshow(obr60_3)
plt.axis("off")
plt.subplot(2, 2, 3)
plt.title("Obrot 60 lewo, expand 0")
plt.imshow(obr60_2)
plt.axis("off")
plt.subplot(2, 2, 4)
plt.title("Obrot 60 prawo, expand 0")
plt.imshow(obr60_4)
plt.axis("off")
plt.savefig("fig_4.png")


im2 = Image.new(mode="RGB", size=(im.width*2, im.height*2), color='red')
im2.paste(im, (int(im2.width/2), int(im2.height/2)))
im2.show()
obr60_5 = im2.rotate(60, expand=1, fillcolor='red', center=(int(im2.width/2), int(im2.height/2)))
obr60_5.save("obrot.png")

t1 = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
t1 = t1.rotate(90, expand=1)
#t1.show()

t1_1 = im.transpose(Image.TRANSPOSE)
#t1_1.show()

t2 = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
t2 = t2.rotate(270, expand=1)
#t2.show()

t2_2 = im.transpose(Image.TRANSVERSE)
#t2_2.show()