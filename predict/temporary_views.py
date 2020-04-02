import os
import cv2
from PLANET.settings import MEDIA_ROOT,BASE_DIR
from django.shortcuts import render
from django.core.cache import cache
from django.core.files import File
from django.core.files.base import ContentFile
from django.http import response,request
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse, reverse_lazy
from django.http.response import StreamingHttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from . import forms
from rest_framework import viewsets, permissions
from .serializers import plantSerializers
from .models import plant, segment
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, model_from_json
from keras.preprocessing.image import img_to_array
from django.contrib import messages
import base64
import io
from PIL import Image
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''apple_model_path = os.path.join(BASE_DIR,'plant_classification_apple_segmented.h5')
apple_model = load_model(apple_model_path, compile = False)'''
# Create your views here.


def home(request):
    return render(request, 'predict/index.html')


def do_segmentation(R_l, B_l, G_l, R_h, G_h, B_h, image,name):
    image_array = np.array(image)
    blurred_frame = cv2.blur(image_array, (5, 5), 0)
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # Defining color theshold
    low_green = np.float32([R_l, G_l, B_l])
    high_green = np.float32([R_h, G_h, B_h])
    # print(low_green,high_green)
    hsv_frame = np.float32(hsv_frame)
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    print("green mask shape = {}".format(green_mask.shape))
    # Morphological adjestments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Getting the largest contour
    contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    try:
        #print(contours)
        biggest = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        print("got biggest contour!")
        cv2.drawContours(image_array, biggest, -1, (0, 0, 0), 1)

        # Creating blank mask and filling in the contour
        blank_mask = np.zeros(image_array.shape, dtype=np.uint8)
        cv2.fillPoly(blank_mask, [biggest], (255, 255, 255))
        blank_mask = cv2.cvtColor(blank_mask, cv2.COLOR_BGR2GRAY)
        print("shape of blank mask = {}".format(blank_mask.shape))
        result = cv2.bitwise_and(image_array, image_array, mask=blank_mask)
        result = np.array(result)
        #x, y, w, h = cv2.boundingRect(blank_mask)
        ROI = result
        # cv2.imshow('image',ROI)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print("roi shape = {}".format(ROI.shape))
        _, buffer_roi = cv2.imencode('.jpg', ROI)
        f_roi = buffer_roi.tobytes()

        f1 = ContentFile(f_roi)     #must send in form of file and for that we need to do this
        image_file = File(f1,name=name)
        #print("f_roi shape = {}".format(f_roi.shape))

        return image_file

    except IndexError:
        print("Index out of range!")
        pass


def predict_plant(image, plant_class):
    image_array = img_to_array(image)
    image_array /= 255
    image_array = np.expand_dims(image_array, axis=0)

    '''model_path = os.path.join(BASE_DIR,'apple.json')
    weight_path = os.path.join(BASE_DIR,'apple.h5')
    with open(model_path,'r') as f:
        plant_model = model_from_json(f.read())'''

    '''model_cache_key = 'model_cache'

    model = cache.get(model_cache_key)'''

    '''if model is None:
        grapes_model_path = os.path.join(BASE_DIR,'plant_classification_segmented_resnet50_grapes.h5')
        plant_model = load_model(grapes_model_path,compile = False)
        target_names = ['Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariealthy']
        cache.set(model_cache_key,plant_model,None) '''

    # plant_model.load_weights(weight_path)
    if plant_class.lower() == 'apple':
        model_cache_key = 'apple_cache'
        plant_model = cache.get(model_cache_key)
        if plant_model is None:
            apple_model_path = os.path.join(
                BASE_DIR, 'apple_classification_segmented_resnet50.h5')
            plant_model = load_model(apple_model_path, compile=False)
            target_names = ['Apple Scab', 'Apple Black Rot',
                            'Apple Cedar Rust', 'Apple Healthy']
            cache.set(model_cache_key, plant_model, None)

    if plant_class.lower() == 'cherry':
        model_cache_key = 'cherry_cache'
        plant_model = cache.get(model_cache_key)
        if plant_model is None:
            cherry_model_path = os.path.join(
                BASE_DIR, 'cherry_classification_segmented_resnet50.h5')
            plant_model = load_model(cherry_model_path, compile=False)
            target_names = [
                'Cherry Powdery Mildew', 'Cherry Healthy']
            cache.set(model_cache_key, plant_model, None)

    if plant_class.lower() == 'corn':
        model_cache_key = 'corn_cache'
        plant_model = cache.get(model_cache_key)
        if plant_model is None:
            corn_model_path = os.path.join(
                BASE_DIR, 'corn_classification_segmented_resnet50.h5')
            plant_model = load_model(corn_model_path, compile=False)
            target_names = ['Corn(maize) Cercospora leaf spot(Gray leaf spot)', 'Corn (maize) Common Rust',
                            'Corn(maize) healthy', 'Corn(maize) Northern Leaf Blight']
            cache.set(model_cache_key, plant_model, None)

    if plant_class.lower() == 'grapes':
        model_cache_key = 'grapes_cache'
        plant_model = cache.get(model_cache_key)
        if plant_model is None:
            grapes_model_path = os.path.join(
                BASE_DIR, 'plant_classification_segmented_resnet50_grapes.h5')
            plant_model = load_model(grapes_model_path, compile=False)
            target_names = [
                'Grape Black Rot', 'Grape Esca(Black_Measles)', 'Grape Leaf blight(Isariopsis_Leaf_Spot)', 'Grape Healthy']
            cache.set(model_cache_key, plant_model, None)

    if plant_class.lower() == 'peach':
        model_cache_key = 'peach_cache'
        plant_model = cache.get(model_cache_key)
        if plant_model is None:
            peach_model_path = os.path.join(
                BASE_DIR, 'plant_classification_peach_segmented.h5')
            plant_model = load_model(peach_model_path, compile=False)
            target_names = ['Peach Bacterial Spot', 'Peach Healthy']
            cache.set(model_cache_key, plant_model, None)

    if plant_class.lower() == 'pepper_bell' or plant_class.lower() == 'pepperbell':
        model_cache_key = 'pepper_bell_cache'
        plant_model = cache.get(model_cache_key)
        if plant_model is None:
            pepper_bell_model_path = os.path.join(
                BASE_DIR, 'plant_classification_pepper_bell_segmented.h5')
            plant_model = load_model(pepper_bell_model_path, compile=False)
            target_names = ['Pepper Bell Bacterial Spot',
                            'Pepper Bell Healthy']
            cache.set(model_cache_key, plant_model, None)

    if plant_class.lower() == 'potato':
        model_cache_key = 'potato_cache'
        plant_model = cache.get(model_cache_key)
        if plant_model is None:
            potato_model_path = os.path.join(
                BASE_DIR, 'plant_classification_potato_segmented.h5')
            plant_model = load_model(potato_model_path, compile=False)
            target_names = ['Potato Early Blight',
                            'Potato Late Blight', 'Potato Healthy']
            cache.set(model_cache_key, plant_model, None)

    if plant_class.lower() == 'strawberry':
        model_cache_key = 'strawberry_cache'
        plant_model = cache.get(model_cache_key)
        if plant_model is None:
            strawberry_model_path = os.path.join(
                BASE_DIR, 'plant_classification_strawberry_segmented.h5')
            plant_model = load_model(strawberry_model_path)
            target_names = ['Strawberry Healthy', 'Strawberry Leaf Scorch']
            cache.set(model_cache_key, plant_model, None)

    if plant_class.lower() == 'tomato':
        model_cache_key = 'tomato_cache'
        plant_model = cache.get(model_cache_key)
        if plant_model is None:
            tomato_model_path = os.path.join(
                BASE_DIR, 'plant_classification_segmented_resnet50_Tomato.h5')
            plant_model = load_model(tomato_model_path)
            target_names = ['Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Healthy', 'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
                            'Tomato Spider mites(Two spotted spider mite)', 'Tomato Target Spot', 'Tomato Mosaic Virus', 'Tomato Yellow Leaf Curl Virus']
            cache.set(model_cache_key, plant_model, None)

    y_hat = plant_model.predict(image_array)
    class_value = np.argmax(y_hat, axis=1)[0]

    return (target_names[class_value], np.max(y_hat)*100)


def formpage(request):
    p_image = plant()
    if request.method == "POST":
        pform = forms.plantForm(request.POST, request.FILES)
        if pform.is_valid():
            plant_class = pform.cleaned_data['species']
            if request.FILES.get("Image", None) is not None:
                plant_image = request.FILES['Image']
                image_bytes = plant_image.read()
                target_image = Image.open(io.BytesIO(image_bytes))
                target_image = target_image.resize((128, 128), Image.ANTIALIAS)
            target_value, prob = predict_plant(target_image, plant_class)
            if 'Image' in request.FILES:
                p_image.species = pform.cleaned_data['species']
                p_image.Image = pform.cleaned_data['Image']
                p_image.save()
            context_dict = {'form': pform, 'disease_value': target_value,
                            'probability': prob, 'plant_show': p_image}
            # return HttpResponse('The predicted class is {}'.format(target_value))
            #messages.success(request,'The predicted class is {}'.format(target_value))
        else:
            print(pform.errors)
    else:
        pform = forms.plantForm()
        context_dict = {'form': pform}
    return render(request, 'predict/predict.html', context=context_dict)


@csrf_exempt
def segment_it(request):
    global segmented_image_showing
    global name_of_image
    global flag
    global image_id

    sform = forms.segmentForm()
    # tform = forms.tempForm()
    s_image = segment()

    if request.method == 'POST':
        # tform = forms.tempForm(request.POST)
        #t_value = request.POST.get('predictIt')
        #print("predict button value = {}".format(pre_val))
        #print(t_value)
        if 'predict' in request.POST:
            apple_model_path = os.path.join(BASE_DIR, 'apple_classification_segmented_resnet50.h5')
            plant_model = load_model(apple_model_path, compile=False)
            target_names = ['Apple_scab', 'Apple Black_rot',
                    'Apple Cedar Rust', 'Apple Healthy']
            s_obj = segment.objects.filter().order_by('-Id')[0]
            image_bytes = s_obj.Image.read()
            target_image = Image.open(io.BytesIO(image_bytes))
            target_image = target_image.resize((128, 128), Image.ANTIALIAS)
            image_array = img_to_array(target_image)
            image_array /= 255
            image_array = np.expand_dims(image_array, axis=0)
            y_hat = plant_model.predict(image_array)
            class_value = np.argmax(y_hat, axis=1)[0]
            disease_value = target_names[class_value]
            context_dict = {'segment_form': sform,'disease_value':disease_value, 'image_show':s_obj}
           
        else:
            sform = forms.segmentForm(request.POST, request.FILES)
            if sform.is_valid():
                if request.FILES.get("Image", None) is not None:
                    plant_image = request.FILES['Image']
                    image_bytes = plant_image.read()
                    name_of_image = request.FILES['Image'].name
                    #name_of_image = request.FILES['Image'].name.split('.')[0] + '_segmented' + '.jpg'
        
                    print(name_of_image)
                    # hue_amount_l = 0
                    # value_amount_l = 36
                    # saturation_amount_l = 10

                    # hue_amount_h = 100
                    # value_amount_h = 255
                    # saturation_amount_h = 255

                    target_image = Image.open(io.BytesIO(image_bytes))
                    segmented_image_showing = target_image.resize((128, 128), Image.ANTIALIAS)
                    #segmented_image_showing = do_segmentation(hue_amount_l, saturation_amount_l, value_amount_l,hue_amount_h, saturation_amount_h, value_amount_h, temp_image)
                    flag = 1
                    s_image.Image = sform.cleaned_data['Image']
                    s_image.save()
                    #img_add = s_image.Image.url
                    s_obj = segment.objects.filter().order_by('-Id')[0]
                    image_id = s_obj.Id
                    print("image id = {}".format(image_id))
                    context_dict = {'segment_form': sform, 'image_show': s_image}
    
    elif request.is_ajax():
        print("ajax one!")
        # High bar values
        hue_amount_h = request.GET.get('h_value_h')
        saturation_amount_h = request.GET.get('s_value_h')
        value_amount_h = request.GET.get('v_value_h')

        # low bar values
        hue_amount_l = request.GET.get('h_value_l')
        saturation_amount_l = request.GET.get('s_value_l')
        value_amount_l = request.GET.get('v_value_l')

        print("high values")
        print('hue : {} \n saturation: {} \n value : {}\n'.format( hue_amount_h, saturation_amount_h, value_amount_h))
        print("low values")
        print('hue : {} \n saturation: {} \n value : {}\n'.format(hue_amount_l, saturation_amount_l, value_amount_l))

        if flag:
            print("changing Image")
            im = do_segmentation(hue_amount_l, saturation_amount_l, value_amount_l,
                                 hue_amount_h, saturation_amount_h, value_amount_h, segmented_image_showing,name_of_image)
            if im:
                # s_obj = segment.objects.filter().order_by('-Id')[0]
                # image_id = s_obj.Id
                s = segment.objects.get(Id=image_id)
                s.Image = im
                s.save()
                img_add = s.Image.url
                return HttpResponse(img_add)
            else:
                print("Image not available")
                sform = forms.segmentForm()
                context_dict = {'segment_form': sform}
        else:
                print("ajax request not maintained properly")
                sform = forms.segmentForm()
                context_dict = {'segment_form': sform}
        
    else:
        sform = forms.segmentForm()
        # tform = forms.tempForm({'predictIt':'no'})
        flag = 0
        context_dict = {'segment_form': sform}
    
    print("final context_dict = {}".format(context_dict))
    return render(request, 'predict/segment.html', context_dict)

class plantViewset(viewsets.ModelViewSet):
    serializer_class = plantSerializers
    queryset = plant.objects.all()
    permission_classes = (permissions.IsAuthenticatedOrReadOnly,)

cap = cv2.VideoCapture(0)
def gen_frame(hh,sh,vh,hl,sl,vl):
    print("taking live photos")
    while True:
        _,frame = cap.read()

        #segmenting it
        image_array = np.array(frame)
        blurred_frame = cv2.blur(image_array, (5, 5), 0)
        hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

        # Defining color theshold
        low_green = np.float32([hl,sl,vl])
        high_green = np.float32([hh,sh,vh])

        print(low_green)
        print(high_green)
        hsv_frame = np.float32(hsv_frame)
        green_mask = cv2.inRange(hsv_frame, low_green, high_green)
        print("green mask shape = {}".format(green_mask.shape))
        # Morphological adjestments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Getting the largest contour
        contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        try:
            #print(contours)
            biggest = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            print("got biggest contour!")
            cv2.drawContours(image_array, biggest, -1, (0, 0, 0), 1)

            # Creating blank mask and filling in the contour
            blank_mask = np.zeros(image_array.shape, dtype=np.uint8)
            cv2.fillPoly(blank_mask, [biggest], (255, 255, 255))
            blank_mask = cv2.cvtColor(blank_mask, cv2.COLOR_BGR2GRAY)
            print("shape of blank mask = {}".format(blank_mask.shape))
            result = cv2.bitwise_and(image_array, image_array, mask=blank_mask)
            result = np.array(result)
            ROI = result
            __, buffer_frame = cv2.imencode('.jpg', ROI)
            f_frame = buffer_frame.tobytes()
            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + f_frame + b'\r\n\r\n')
        except IndexError:
            print("Index out of range!")
global hh,sh,vh,hl,sl,vl
hh = 100
sh = 255
vh = 255
hl = 0
sl = 10
vl = 36

@csrf_exempt
def streaming_live(request):
    hh = 100
    sh = 255
    vh = 255
    hl = 0
    sl = 10
    vl = 36
    if request.is_ajax():
        print("ajax one!")
        # High bar values
        hh = request.GET.get('h_value_h')
        sh = request.GET.get('s_value_h')
        vh = request.GET.get('v_value_h')

        # low bar values
        hl = request.GET.get('h_value_l')
        sl = request.GET.get('s_value_l')
        vl = request.GET.get('v_value_l')
        return StreamingHttpResponse(gen_frame(hh,sh,vh,hl,sl,vl),content_type='multipart/x-mixed-replace; boundary=frame') 
    else:
        return StreamingHttpResponse(gen_frame(hh,sh,vh,hl,sl,vl),content_type='multipart/x-mixed-replace; boundary=frame')

def button_segment_live(request) :
    submitbutton = request.POST.get('Submit')
    if submitbutton:
        context = {'submitbutton' : submitbutton}
    else:
        context = {'submitbutton' : None}
    return render(request, 'predict/live_segment.html', context) 
