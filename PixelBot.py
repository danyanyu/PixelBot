import telebot
from telebot import types
import os
#import logging
#logger = logging.getLogger(__name__)

import numpy as np
from numba import prange, njit
from skimage import io
import matplotlib.pyplot as plt

from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, sRGBColor

import requests



#image processing part
gausmatrix = np.array([[2,4,5,4,2],
                       [4,9,12,9,4],
                       [5,12,15,12,5],
                       [4,9,12,9,4],
                       [2,4,5,4,2]])/159

@njit
def togray1d(p):
    c = np.sqrt((p[:,:,0]/256)**2 + (p[:,:,1]/256)**2 + (p[:,:,2]/256)**2)
    return c

@njit
def togray(p):
    x,y,_ =p.shape
    k=np.zeros((x,y,3),dtype='uint8')
    c = (p[:,:,0]+p[:,:,1]+p[:,:,2])//3
    k[:,:,0]=c
    k[:,:,1]=c
    k[:,:,2]=c
    return k

@njit
def gausblur5(p):
    c = p.copy()
    first, second = c.shape
    for i in prange(2, first-2):
        for j in range(2, second-2):
            c[i, j] = np.average(c[i-2:i+3,j-2:j+3],weights=gausmatrix)
    return c

@njit
def gausblurfast(p):
    gaus1d = np.exp(-0.5 * (np.arange(-2, 3)) ** 2) / np.sqrt(2 * np.pi)
    c = p.copy()
    first, second = c.shape
    for i in prange(2, first-2):
        for j in range(2, second-2):
            c[i, j] = np.average(c[i-2:i+3,j],weights=gaus1d)
    for i in prange(2, first-2):
        for j in range(2, second-2):
            c[i, j] = np.average(c[i,j-2:j+3],weights=gaus1d)
    return c

#edge detection stuff
@njit
def gx(p):
    c = np.zeros_like(p)
    first, second = c.shape
    coef = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    for i in prange(1, first-1):
        for j in range(1, second-1):
            c[i,j] += 1*(p[i-1,j-1]+p[i+1,j-1])
            c[i,j] += -1*(p[i-1,j+1]+p[i+1,j+1])
            c[i,j] += 2*p[i,j-1]
            c[i,j] += -2*p[i,j+1]
    return c

@njit
def gy(p):
    c = np.zeros_like(p)
    first, second = c.shape
    coef = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    for i in prange(1, first-1):
        for j in range(1, second-1):
            c[i,j] += 1*(p[i-1,j-1]+p[i-1,j+1])
            c[i,j] += -1*(p[i+1,j-1]+p[i+1,j+1])
            c[i,j] += 2*p[i-1,j]
            c[i,j] += -2*p[i+1,j]
    return c

@njit
def thresh(p):
    c = np.arctan2(gy(p),gx(p))
    m = np.sqrt(gx(p)**2 + gy(p)**2)
    k = np.zeros_like(p)
    first, second = c.shape
    for i in prange(1,first-1):
        for j in range(1,second-1):
            if np.pi/(-8)<=c[i,j]<=np.pi/8 or c[i,j]>=np.pi*7/8 or c[i,j]<=np.pi*7/(-8):
                if m[i,j-1]<=m[i,j] and m[i,j]>=m[i,j+1]:#0
                    k[i,j]=m[i,j]
            elif np.pi/8<=c[i,j]<=np.pi*3/8 or np.pi*5/(-8)>=c[i,j]>=np.pi*7/(-8):
                if m[i-1,j+1]<=m[i,j] and m[i,j]>=m[i+1,j-1]:#45
                    k[i,j]=m[i,j]
            elif np.pi*3/8<=c[i,j]<=np.pi*5/8 or np.pi*3/8<=(-1)*c[i,j]<=np.pi*5/8:
                if m[i-1,j]<=m[i,j] and m[i,j]>=m[i+1,j]:#90
                    k[i,j]=m[i,j]
            else:
                if m[i-1,j-1]<=m[i,j] and m[i,j]>=m[i+1,j+1]:#135
                    k[i,j]=m[i,j]
    return k

@njit
def dfresh(p, low=0.05,high=0.08):
    weak = np.zeros_like(p)
    strong = np.zeros_like(p)
    first, second = p.shape
    m = np.max(p)
    for i in prange(first):
        for j in range(second):
            if p[i,j] >= high*m:
                strong[i,j]=1
            elif p[i,j] >=low*m:
                weak[i,j]=1
    for i in prange(1,first-1):
        for j in range(1,second-1):
            if weak[i,j] > 0:
                if np.sum(strong[i-1:i+2,j-1:j+2])==0:
                    weak[i,j]=0
    return (strong + weak)*(-1)+1

noiseLoaded = False
noise = None

def stippling(p):
    global noiseLoaded
    global noise
    if noiseLoaded == False:
        noise = io.imread("noise.png")[:, :, 0] / 255
        noiseLoaded = True

    c = np.zeros_like(p)
    yd, xd = p.shape
    xn = xd//32
    yn = yd//32
    xr = xd%32
    yr = yd%32
    for i in range(yn):
        for j in range(xn):
            for ii in range(32):
                for jj in range(32):
                    if p[i*32+ii,j*32+jj]>noise[ii,jj]:
                        c[i*32+ii,j*32+jj] = 1
    for j in range(xn):
        for ii in range(yr):
            for jj in range(32):
                if p[yn*32+ii,j*32+jj]>noise[ii,jj]:
                    c[yn*32+ii,j*32+jj] = 1
    for i in range(yn):
        for jj in range(xr):
            for ii in range(32):
                if p[i*32+ii,xn*32+jj]>noise[ii,jj]:
                    c[i*32+ii,xn*32+jj] = 1
    for ii in range(yr):
        for jj in range(xr):
            if p[yn*32+ii,xn*32+jj]>noise[ii,jj]:
                c[yn*32+ii,xn*32+jj] = 1
    return c

def combine(p1,edges):
    res = np.ones_like(p1)
    first, second = p1.shape
    for i in range(first):
        for j in range(second):
            if edges[i,j]>0:
                res[i,j]=p1[i,j]
            else:
                res[i,j]=0
    return res


def process_img_broken(pic):#, save=False, savename=None):
    #p = io.imread("images/" + filename)
    p = pic.copy()
    ress = pic.copy()
    p = togray1d(p)
    orig = p.copy()
    p = gausblurfast(p)
    p = np.sqrt(gx(p) ** 2 + gy(p) ** 2) * (-1)
    p = dfresh(thresh(p) * (-1))
    orig = stippling(orig)
    res = 255*gausblurfast(combine(orig, p * (-1) + 1.))
    ress[:, :, 0] = res.astype('uint8')[:,:]
    ress[:, :, 1] = res.astype('uint8')[:,:]
    ress[:, :, 2] = res.astype('uint8')[:,:]
    return ress
    #return ress
    #return res
    #if save:
    #	io.imsave("images/inktober/edits/" + filename, res)


def process_img_drawing(pic):#filename, save=False, savename=None, togr=True):
    #p = io.imread("images/" + filename)
    p = pic.copy()
    p = togray1d(p)
    orig = p.copy()
    p = gausblurfast(p)
    p = np.sqrt(gx(p) ** 2 + gy(p) ** 2)
    p = dfresh(thresh(p))
    orig = stippling(orig)
    res = combine(orig, p)
    res *= 255

    ress = pic.copy()
    ress[:, :, 0] = res.astype('uint8')[:,:]
    ress[:, :, 1] = res.astype('uint8')[:,:]
    ress[:, :, 2] = res.astype('uint8')[:,:]
    return ress
    #return ress
    #fig = plt.figure(tight_layout=True, figsize=(9, 9))
    #plt.imshow(res, cmap='gray')
    #if save:
    #	io.imsave("images/inktober_edits/" + savename + ".png", res)

#end edge detection stuff

@njit
def pixelate(pic, psize=16):
    p=pic.copy()
    x, y, _ = p.shape
    nx=x//psize
    ny=y//psize
    rx=x%psize
    ry=y%psize
    for i in prange(nx):
        for j in range(ny):
            tmp = np.zeros(3)
            for ii in range(psize):
                for jj in range(psize):
                    tmp += p[i*psize+ii,j*psize+jj,:3]
            tmp//=psize*psize
            p[i*psize:(i+1)*psize,j*psize:(j+1)*psize,:3]=tmp
    for j in prange(ny):
        tmp = np.zeros(3)
        for i in range(rx):
            for jj in range(psize):
                tmp += p[nx*psize+i,j*psize+jj,:3]
        tmp//=psize*rx
        p[nx*psize:,j*psize:(j+1)*psize,:3]=tmp
    for i in prange(nx):
        tmp = np.zeros(3)
        for j in range(ry):
            for ii in range(psize):
                tmp += p[i*psize+ii,ny*psize+j,:3]
        tmp//=psize*ry
        p[i*psize:(i+1)*psize,ny*psize:,:3]=tmp
    tmp = np.zeros(3)
    for j in prange(ry):
        for i in range(rx):
            tmp += p[nx*psize+i,ny*psize+j,:3]
    tmp//=rx*ry
    p[nx*psize:,ny*psize:,:3]=tmp
    return p, nx, ny


@njit
def upscale(p, factor=2):
    x, y,z = p.shape
    res = np.zeros((factor*x,factor*y,z),dtype="uint8")
    for i in prange(x):
        for j in range(y):
            res[factor*i:factor*(i+1),factor*j:factor*(j+1),:]+=p[i,j,:z]
    return res

@njit
def downscale(p, factor=2):
    x, y,z = p.shape
    xn = x//factor
    yn = y//factor
    res = np.zeros((xn,yn,z), dtype='uint8')
    for i in prange(xn):
        for j in range(yn):
            res[i,j,:]=p[i*factor,j*factor,:]
    return res


def imgtolab(p):
    x,y,_=p.shape
    k = np.zeros((x,y,3))
    for i in prange(x):
        for j in range(y):
            rgb = sRGBColor(*(p[i,j,:3]/255.0))
            k[i,j,:] = np.array(convert_color(rgb, LabColor).get_value_tuple())
    return k

def imgtorgb(p):
    x,y,_=p.shape
    k = np.zeros((x,y,3))
    for i in prange(x):
        for j in range(y):
            lab = LabColor(*(p[i,j,:3]))
            k[i,j,:] = np.array(convert_color(lab, sRGBColor).get_value_tuple())
    return k


@njit
def shrinkcolcount(p,palette):
    x,y,_=p.shape
    k = p.copy()
    for i in prange(x):
        for j in range(y):
            df = palette[0][:3] - p[i,j,:]
            et = df @ df
            col = palette[0]
            for pal in palette:
                df = pal[:3] - p[i,j,:]
                ddf = df@df
                if ddf < et:
                    et = ddf
                    col = pal
            k[i,j,:3]=col[:3]
    return k, len(palette)


@njit
def shrinkcolcountrgb(p,palette):
    #dummy df calculated without LAB conversion
    x,y,_=p.shape
    k = p.copy()
    for i in prange(x):
        for j in range(y):
            rmean=0.5*(palette[0,0]+p[i,j,0])
            etsq = (2+rmean/256)*((palette[0,0]-p[i,j,0])**2)+4*((palette[0,1]-p[i,j,1])**2)+(2+(255-rmean)/256)*((palette[0,2]-p[i,j,2])**2)
            col = palette[0]
            for pal in palette:
                rmean=0.5*(pal[0]+p[i,j,0])
                df = (2+rmean/256)*((pal[0]-p[i,j,0])**2)+4*((pal[1]-p[i,j,1])**2)+(2+(255-rmean)/256)*((pal[2]-p[i,j,2])**2)
                if df < etsq:
                    etsq = df
                    col = pal
            k[i,j,:3]=col[:3]
    return k, len(palette)


def applypalette(pic, palette, size=2):#, save=False):
    p=pic.copy()
    #p = io.imread("images/"+filename)
    p = downscale(pixelate(p,psize=size)[0],factor=size)
    p = imgtolab(p)
    #fig = plt.figure(tight_layout=True, figsize=(9,9))
    p = upscale(imgtorgb(shrinkcolcount(p,palette)[0]),factor=size)
    #plt.imshow(p)
    #if save:
    #    io.imsave("images/pixels/"+filename+".png",p)
    return p


def loadpalette(filename, lab=True):
    palette = io.imread("images/palettes/"+filename)
    if lab:
        palette = imgtolab(palette)
    return palette


def applypalettergb(pic, palettename, size=2):#, save=False):
    #p = io.imread("images/"+filename)
    p=pic.copy()
    palette = io.imread("palettes/"+palettename+".png")[0]
    p = downscale(pixelate(p,psize=size)[0],factor=size)
    p = upscale(shrinkcolcountrgb(p,palette)[0],factor=size)
    #if save:
    #    io.imsave("images/pixels/"+filename+".png",p)
    return p


@njit
def bayer4x4(p,N):
    bayermatrix = np.array([[0,8,2,10],
                            [12,4,14,6],
                            [3,11,1,9],
                            [15,7,13,5]])
    #bayermatrix=np.array([[0,2],[3,1]])
    res = p.copy()
    x,y,_=p.shape
    r = 255//N
    for i in prange(x):
        for j in range(y):
            bnoise = (bayermatrix[i%4,j%4]/16-0.5)*(1/N)
            res[i,j,0]= 255*np.floor((res[i,j,0]/255 + bnoise)*(N-1)+0.5)/(N-1)
            res[i,j,1]= 255*np.floor((res[i,j,1]/255 + bnoise)*(N-1)+0.5)/(N-1)
            res[i,j,2]= 255*np.floor((res[i,j,2]/255 + bnoise)*(N-1)+0.5)/(N-1)
    return res

@njit
def bayer4x4bw(p):
    bayermatrix = np.array([[0,8,2,10],
                            [12,4,14,6],
                            [3,11,1,9],
                            [15,7,13,5]])
    #bayermatrix=np.array([[0,2],[3,1]])
    res = p.copy()
    x,y,_=p.shape
    for i in prange(x):
        for j in range(y):
            bnoise = (bayermatrix[i%4,j%4]/16-0.5)*0.5
            res[i,j,0]= 255*np.floor((res[i,j,0]/255 + bnoise)+0.5)
            res[i,j,1]= 255*np.floor((res[i,j,1]/255 + bnoise)+0.5)
            res[i,j,2]= 255*np.floor((res[i,j,2]/255 + bnoise)+0.5)
    return res


#end image processing part

#bot part

user_choices = {}

sizes = ['1x1', '2x2', '4x4', '8x8', '16x16', '32x32']
modes = ['nd', 'pixel', 'drawing', 'drawing_broken']
palettes = ['none (Bayer4x4 algorithm)', 'aap', 'apollo', 'blessing', 'endesga64',
            'normal71', 'nebula', 'nyx8', 'normal32', 'normal36', 'pastel',
            'picotron', 'pollen', 'robot', 'rocca', 'slynyrd128', 'softserve', 'twilight', 'yamazaki']

preview = {}

bot = telebot.TeleBot("5802386456:AAG73F64Z0hathHLSF8QYeud-UVnubCpuFM", parse_mode=None) # You can set parse_mode by default. HTML or MARKDOWN


@bot.message_handler(func=lambda message: message.from_user.id not in user_choices)
def check(message):
    global user_choices, preview
    user_choices[message.from_user.id] = ['0']
    preview[message.from_user.id] = False
    bot.send_message(message.from_user.id, "Welcome to PixelBot. "
                                               "I can apply effects to your images and make them look like a pixel art."
                                           " You can also try drawing-like effects, inspired by Acerola's YT videos."
                                           " Formats such as PNG and JPG work best, consider converting your files beforehand.")

@bot.message_handler(commands=['start'])
def send_welcome(message):
    print(f'{message.from_user.first_name} started bot')
    global preview, user_choices
    user_choices[message.from_user.id] = ['0']
    preview[message.from_user.id] = False
    bot.send_message(message.from_user.id, "Welcome to PixelBot. "
                                           "I can apply effects to your images and make them look like a pixel art."
                                           " You can also try drawing-like effects, inspired by Acerola's YT videos."
                                           " Formats such as PNG and JPG work best, consider converting your files beforehand.")


@bot.message_handler(commands=['help'])
def send_help(message):
    print(f'{message.from_user.first_name} asked for help')
    bot.send_message(message.from_user.id, "Under construction...")


@bot.message_handler(commands=['drawing'])
def send_drawing(message):
    markup = types.ReplyKeyboardRemove(selective=False)
    print(f'{message.from_user.first_name} chose drawing')
    global user_choices
    user_choices[message.from_user.id] = ['2']
    bot.send_message(message.from_user.id, "Send me a photo as a file.", reply_markup=markup)



@bot.message_handler(commands=['brokendrawing'])
def send_brokendrawing(message):
    markup = types.ReplyKeyboardRemove(selective=False)
    print(f'{message.from_user.first_name} chose brokendrawing')
    global user_choices
    user_choices[message.from_user.id] = ['3']
    bot.send_message(message.from_user.id, "Send me a photo as a file.", reply_markup=markup)


@bot.message_handler(func=lambda message: user_choices[message.from_user.id][0] ==
                                          '1' and message.content_type == 'text' and message.text in palettes and preview[message.from_user.id])
def show_preview(message):
    global user_choices, preview
    if message.text == 'none (Bayer4x4 algorithm)':
        print(f'{message.from_user.first_name} chose palette: ' + message.text)
        markup = types.ReplyKeyboardMarkup(row_width=1)
        preview[message.from_user.id] = False
        user_choices[message.from_user.id].append(message.text)
        for i in range(len(sizes)):
            markup.row(types.KeyboardButton(sizes[i]))
        bot.send_message(message.from_user.id, "Choose the size of new pixels:", reply_markup=markup)
    else:
        print(f'{message.from_user.first_name} looked at preview')
        markup = types.ReplyKeyboardMarkup(row_width=1)
        preview[message.from_user.id] = False
        user_choices[message.from_user.id].append(message.text)
        markup.row(types.KeyboardButton('This one!'))
        markup.row(types.KeyboardButton('Back to choosing.'))
        bot.send_message(message.from_user.id, "This is a preview of chosen palette:", reply_markup=markup)
        p = io.imread('palettes/'+message.text+'.png')
        pr = upscale(p, factor=10)
        io.imsave(message.text+'preview.jpg', pr)
        photo = open(message.text+'preview.jpg', 'rb')
        bot.send_photo(message.chat.id, photo)
        photo.close()
        os.remove(message.text+'preview.jpg')


@bot.message_handler(func=lambda message: user_choices[message.from_user.id][0] ==
                                          '1' and message.content_type == 'text' and message.text ==
                                          'This one!' and len(user_choices[message.from_user.id])==2 and preview[message.from_user.id] == False)
def send_palettechoice(message):
    print(f'{message.from_user.first_name} chose palette: '+message.text)
    markup = types.ReplyKeyboardMarkup(row_width=1)
    for i in range(len(sizes)):
        markup.row(types.KeyboardButton(sizes[i]))
    bot.send_message(message.from_user.id, "Choose the size of new pixels:", reply_markup=markup)


@bot.message_handler(func=lambda message: user_choices[message.from_user.id][0] ==
                                          '1' and message.content_type == 'text' and message.text == 'Back to choosing.' and preview[message.from_user.id] == False)
def send_palettechoice(message):
    print(f'{message.from_user.first_name} back to choosing')
    global preview
    preview[message.from_user.id] = True
    global user_choices
    user_choices[message.from_user.id] = ['1']
    markup = types.ReplyKeyboardMarkup(row_width=1)
    for i in range(len(palettes)):
        markup.row(types.KeyboardButton(palettes[i]))
    bot.send_message(message.from_user.id, "Choose a specific colour "
                                           "palette from below or a standard color quantization algorithm (none):",
                     reply_markup=markup)


@bot.message_handler(func=lambda message: user_choices[message.from_user.id][0] == '1' and len(
    user_choices[message.from_user.id]) == 2 and message.content_type == 'text' and message.text in sizes)
def send_sizechoice(message):
    print(f'{message.from_user.first_name} chose size: '+message.text)
    markup = types.ReplyKeyboardRemove(selective=False)
    global user_choices
    user_choices[message.from_user.id].append(message.text)
    if user_choices[message.from_user.id][1] == 'none (Bayer4x4 algorithm)':
        bot.send_message(message.from_user.id, "Type an integer for the number of colors you want (the resulting image will contain three times more different colors):", reply_markup=markup)
    else:
        bot.send_message(message.from_user.id, "Send me a photo as a file.", reply_markup=markup)


@bot.message_handler(func=lambda message: len(user_choices[message.from_user.id]) == 3, regexp="[0-9]+")
def send_colornumberchoice(message):
    print(f'{message.from_user.first_name} chose number of colors: '+message.text)
    markup = types.ReplyKeyboardRemove(selective=False)
    global user_choices
    user_choices[message.from_user.id].append(message.text)
    bot.send_message(message.from_user.id, "Send me a photo as a file.", reply_markup=markup)


@bot.message_handler(commands=['pixel'])
def send_options(message):
    print(f'{message.from_user.first_name} chose pixel')
    global preview
    preview[message.from_user.id] = True
    global user_choices
    user_choices[message.from_user.id] = ['1']
    markup = types.ReplyKeyboardMarkup(row_width=1)
    for i in range(len(palettes)):
        markup.row(types.KeyboardButton(palettes[i]))
    bot.send_message(message.from_user.id, "Choose a specific colour "
                                           "palette from below or a standard color quantization algorithm (none):", reply_markup=markup)


@bot.message_handler(content_types=['document'])
def process_image(message):
    markup = types.ReplyKeyboardRemove(selective=False)
    print(f'{message.from_user.first_name} processed an image')
    file_id = message.document.file_id
    file_info = bot.get_file(file_id)

    #file = requests.get('https://api.telegram.org/file/bot{0}/{1}'.format("5802386456:AAG73F64Z0hathHLSF8QYeud-UVnubCpuFM", file_info.file_path))
    bot.send_message(message.from_user.id, "Processing image...", reply_markup=markup)
    p = io.imread('https://api.telegram.org/file/bot{0}/{1}'.format("5802386456:AAG73F64Z0hathHLSF8QYeud-UVnubCpuFM", file_info.file_path))
    #io.imshow(p)
    done = False
    res = np.zeros_like(p)
    global user_choices
    configuration = user_choices[message.from_user.id]
    if configuration[0] == '0':
        bot.send_message(message.from_user.id, "Failed: no processing option selected.\nTry using pixel or drawing commands.")
    elif configuration[0] == '1':
        psize = 0
        if configuration[2] == '1x1':
            psize = 1
        elif configuration[2] == '2x2':
            psize = 2
        elif configuration[2] == '4x4':
            psize = 4
        elif configuration[2] == '8x8':
            psize = 8
        elif configuration[2] == '16x16':
            psize = 16
        elif configuration[2] == '32x32':
            psize = 32

        if configuration[1] == 'none (Bayer4x4 algorithm)':
            if len(configuration) > 3:
                res = upscale(bayer4x4(downscale(p, psize), int(configuration[3])),psize)
                done = True
            else:
                bot.send_message(message.from_user.id,
                                 "Failed: number of colors not specified.")
                user_choices[message.from_user.id] = ['0']
        else:
            res = applypalettergb(p, configuration[1], psize)
            done = True
    elif configuration[0] == '2':
        res = process_img_drawing(p)
        done = True
    else:
        #mode = 3
        res = process_img_broken(p)
        done = True
    if done:
        io.imsave("tmpfiles/tmp"+str(file_id)+".jpg", res)
        photo = open("tmpfiles/tmp"+str(file_id)+".jpg", 'rb')
        bot.send_document(message.chat.id, photo)
        photo.close()
        os.remove("tmpfiles/tmp"+str(file_id)+".jpg")
    user_choices[message.from_user.id] = ['0']


@bot.message_handler(content_types=['photo'])
def hint(message):
    markup = types.ReplyKeyboardRemove(selective=False)
    print(f'{message.from_user.first_name} sent photo instead of file')
    bot.send_message(message.from_user.id, "Please, send your photo as a file.", reply_markup=markup)


bot.infinity_polling()