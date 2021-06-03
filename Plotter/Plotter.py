#Created 12.03.2020 by Ole Oggesen

import numpy as np
import matplotlib.pyplot as plt
import os
import array
import xml.etree.ElementTree as ET
import scipy.signal as sig
from scipy.signal import butter, lfilter


def plot(xarray, yarray, title, xlabel, ylabel,  save, labelarray = None, xlim = None, ylim = None, savePath = None, grid=False):
    '''Function for plotting the measurments on the NMR MRT
    y can contain multiple dataarrays,
    save is either True or False
    xlim is a touple with 2 values
    Needs a plots folder in same folder to save the plots'''
    plt.figure()
    for x, y in zip(xarray, yarray):
        plt.plot(x, y, linewidth=1)
    #plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if labelarray:
        plt.legend(labelarray)
    if grid:
        plt.grid()
    if xlim is not None:
        if isinstance(xlim, type(())):
            plt.xlim(xlim)
    if ylim is not None:
        if isinstance(ylim, type(())):
            plt.ylim(ylim)
    if save == True:
        if savePath is None:
            savetitle = "./plots/" + title.replace(" ", "_") + ".pdf"
        else:
            savePath = savePath + "/plots"
            if not os.path.exists(savePath):
                os.mkdir(savePath)
            savetitle = savePath + "/" + title.replace(" ", "_") + ".pdf"
        plt.savefig(savetitle, format="pdf")
        plt.close()
    else:
        plt.show()

def load(path):
    '''Loads data
    supports datatypes .bin and .csv'''
    if ".bin" in path:
        with open(path, "rb") as file:
            data = file.read()
            if data[0:5] == b'<?xml':
                tree = ET.parse(path)
                data = tree.getroot()
            else:
                data = array.array("f", data)
            return data
    elif ".csv" in path:
        with open(path, 'r') as f:
            data = f.read()
            data = data.split("\n")
            try:
                data.remove("")
            finally:
                assert isinstance(data, list)
                return data

def extractData(path):
    """loads the data from the .Wfm file
     and parses all data needed for plotting
      in status from the casual file. Only compatible with .csv and .bin files
      :returns data, status"""

    try:
        data = load(path + ".Wfm.csv")
        args = load(path + ".csv")
    except FileNotFoundError:
        try:
            data = load(path + ".Wfm.bin")
            args = load(path + ".bin")
        except FileNotFoundError:
            raise FileNotFoundError("File not found. Only .bin and .csv datatypes are supported.")
    status = {}

    # parse for x statuses
    if isinstance(args, type([])):
        for arg in args:
            if "BaseUnit:" in arg:
                tmp = arg.split(":")
                assert tmp[1] == "V"
            if "HardwareXStart" in arg:
                tmp = arg.split(":")
                status["HardwareXStart"] = float(tmp[1])
            elif "HardwareXStop" in arg:
                tmp = arg.split(":")
                status["HardwareXStop"] = float(tmp[1])

        #parse for y statuses
        if ";" in data[0]:
            tmp = []
            for i, point in enumerate(data):
                data_split = point.split(";")
                while len(tmp) < len(data_split):
                    tmp.append([])
                for j, d in enumerate(data_split):
                    tmp[j].append(float(d))
            data = tmp
        else:
            for i, point in enumerate(data):
                data[i] = float(point)
            data = [data]

    if isinstance(args, ET.Element):
        for child in args:
            for kids in child:
                attribute = kids.attrib
                if "HardwareXStart" in attribute["Name"]:
                    status["HardwareXStart"] = float(attribute["Value"])
                if "HardwareXStop" in attribute["Name"]:
                    status["HardwareXStop"] = float(attribute["Value"])
                if "SignalHardwareRecordLength" in attribute["Name"]:
                    #decodes how many signals were recorded and seperates them
                    numOfSignals = int(len(data)/int(attribute["Value"]))
                    tmp = []
                    for i in range(numOfSignals):
                        tmp.append(data[i::numOfSignals])
                    data = tmp

    assert len(status) == 2
    return data,  status


def Plotter(paths, title, save=False, labelarray = None, xlim = None, extraData = None, multiply=None):
    """Plots all data from given path automatically, which are stored in .csv or .bin format.
    extraData for extra plots will always be plotted at last"""
    yarray = []
    xarray = []

    for path in paths:
        data, status = extractData(path)

        for dataElem in data:
            if xlim is not None:
                dataElem, x, _ = RectWindow(dataElem, xlim, status)
            else:
                x = np.linspace(status["HardwareXStart"], status["HardwareXStop"], len(dataElem))
            if save is True:
                maxlength = 5000
            else:
                maxlength = len(dataElem)
            if len(dataElem) > maxlength:
                x, dataElem = downSample(data=dataElem, maxlength=maxlength, x=x)
            yarray.append(list(dataElem))
            xarray.append(list(x))
    if extraData is not None:
        xarray.append(list(extraData[0]))
        yarray.append(list(extraData[1]))
    if multiply is not None:
        for i, y in enumerate(yarray):
            for j, elem in enumerate(y):
                yarray[i][j] = yarray[i][j] * multiply[i]
    counter = 0
    for i in range(len(xarray)):
        while np.max(xarray[i]) < 0.1 and np.min(xarray[i]) > -0.1:
            for k in range(len(xarray)):
                for j in range(len(xarray[k])):
                    xarray[k][j] = xarray[k][j] * 1000
            if xlim is not None:
                xlim = (xlim[0] * 1000, xlim[1] * 1000)
            counter = counter + 1
        xlabellist = ["s", "ms", "us", "ns", "ps"]
        xlabel = "time in " + xlabellist[counter]

    try:
        assert len(xarray) != 0 and len(yarray) != 0
    except:
        raise AssertionError("This file in contains no information. An empty list was created.")
    ylim = (np.min(yarray[0])*1.1, np.max(yarray[0])*1.1)


    plot(xarray, yarray, title, xlabel, "voltage in V", save, labelarray=labelarray, xlim=xlim, savePath=os.path.dirname(path), ylim=ylim)

def findPaths(path, search_key=None):
    """Finds the path of data in a given folder and its subfolders. If search_key is given, only paths which contain the search_key are returned.
    Else it returns all Wfm.bin and Wfm.csv files and excludes its endings."""
    paths = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_dir():
                try:
                    paths.extend(findPaths(entry.path, search_key=search_key))
                except:
                    pass
            elif entry.is_file():
                if search_key is not None:
                    if search_key in entry.name:
                        paths.append(str(entry.path))
                else:
                    if ".Wfm.csv" in entry.name:
                        tmp = str(entry.path).replace(".Wfm.csv", "")
                        paths.append(tmp)
                    elif ".Wfm.bin" in entry.name:
                        tmp = str(entry.path).replace(".Wfm.bin", "")
                        paths.append(tmp)
    if paths == []:
        pass
    else:
        return paths


def Autoplot(path, titledict = None):
    """Plots all .bin and .csv saved data in given folder and all subfolders."""
    paths = findPaths(path)
    for path in paths:
        title = None
        if titledict is not None:
            for key in titledict.keys():
                if key in path:
                    title = titledict[key]
                    break
        if title == None:
            title = path.split("/")
            title = title[-1]
        Plotter([path], title, True,)


def FFTPlot(path, title=None, xlim=None, save=False, freqLim=None):
    """Plots a FFT from the data in given path."""
    data, status = extractData(path)

    if xlim is not None:
        data, x, step = RectWindow(data[0], xlim, status)
        x = None
    else:
        data = data[0]  #TODO adapt for multiinput signals
        step = (status["HardwareXStop"] - status["HardwareXStart"]) / len(data)
    Tmax=step

    mag, freq = FFT(data, step, Tmax)
    data = None

    for i in range(len(freq)):
        freq[i] = freq[i] * 1e-6
    plt.figure()
    plt.stem(freq, mag, use_line_collection=True)
    plt.xlabel("frequency in MHz")
    plt.ylabel("amplitude in V")
    if freqLim is not None:
        freqLim = (freqLim[0]*1e-6, freqLim[1]*1e-6)
        plt.xlim(freqLim)
    if title is not None:
        plt.title(title)
    if save is True:
        saveTitle = os.path.dirname(path) + "/plots"
        if not os.path.exists(saveTitle):
            os.mkdir(saveTitle)
        if title is not None:
            saveTitle = saveTitle + "/" + title.replace(" ", "_") + "_FFT.pdf"
        else:
            tmp = path.plit("/")
            saveTitle = saveTitle + "/" + tmp[-1] +"_FFT.pdf"
        plt.savefig(saveTitle, format="pdf")
        plt.close("all")
    else:
        plt.show()


def RectWindow(dataElem, xlim, status):
    """
    Muliplies the data in dataElem with a rectangularlar window.
    :param dataElem: array like containing information
    :param xlim: touple with start and end point
    :param status:
    :return:
    """
    step = (status["HardwareXStop"] - status["HardwareXStart"]) / len(dataElem)
    dataElem = dataElem[int((xlim[0] - status["HardwareXStart"]) / step):int((xlim[1] - status["HardwareXStart"]) / step)]
    x = np.linspace(xlim[0], xlim[1], len(dataElem))
    try:
        assert len(dataElem) != 0
    except AssertionError:
        raise AssertionError("Wrong xlim data input. There is no data for given time period.")
    step = (xlim[1] - xlim[0]) / len(dataElem)
    return dataElem, x, step

def downSample(data, maxlength, x=None):
    """
    Downsampling.
    :param data: array like
    :param maxlength: maximum length of data array like structure
    :param x: Same as data for two dimensional information
    :return:
    """
    data = data[0::int(len(data) / maxlength)]
    if x is not None:
        x = x[0::int(len(x) / maxlength)]
        return x, data
    return data

def getData(path, xlim=None):
    data, status = extractData(path)

    if xlim is not None:
        data, x, _ = RectWindow(data[0], xlim, status)
    return data

def calcPower(data):
    #Calculates power over a 50 Ohms resistor
    return np.sum(np.array(data)**2)/50/len(data)

def measureFreq(path, xlim=None, freqLim = None):
    """Makes a FFT of the received data, and writes the dominant frequency to an .txt file."""
    data, status = extractData(path)

    for i, d in enumerate(data):
        if xlim is not None:
            d, t, step = RectWindow(dataElem=d, xlim=xlim, status=status)
        else:
            step = (status["HardwareXStop"] - status["HardwareXStart"]) / len(d)
        T0 = step * len(d)
        T0min = 1/20    #Defines the sampling rate in the frequency domain: fs = 1/T=min TODO set bigger value if final run
        if T0min > T0:
            d.extend(np.zeros(int(T0min / step) - int(T0 / step)))      #Zero padding
        mag, freq = FFT(data=d, step=step)
        freq = freq[0:int(len(freq)/2)]
        if freqLim is None:
            freqLim = (3e6, 5e6)
        idx_start = binarySearch(freq, freqLim[0])
        idx_stop = binarySearch(freq, freqLim[1])

        m_max = 0.0
        f_max = 0.0
        for f, m in zip(freq[idx_start:idx_stop], mag[idx_start:idx_stop]):
            if m>m_max:
                f_max = f
                m_max = m



        file_path = os.path.dirname(path)
        with open(file_path + "/freqChannel{}.txt".format(i+1), "a") as file:
            file.write("{}\n".format(f_max))

def measureFreqAuto(path, xlim = None):
    """Measures the dominant frequency in the data stored in path address and saves it to a .txt file. Only frequencies between 3MhZ and 5 MhZ are evaluated.
    :param path: path and subfolders are searched for measuerment data
    :param xlim: (t_start, t_stop) Makes a rectangular window over given time period between t_start and t_stop
    :return: .txt file containing the frequency of highest amplitude between 3 MHz and 5 MHz"""
    paths = findPaths(path)
    for path in paths:  #Removes the stored data in .txt files if they exist
        path = os.path.dirname(path)
        for i in range(2):
            data = path + "/freqChannel{}.txt".format(i + 1)
            if os.path.exists(data):
                os.remove(data)
    for path in paths:
        if xlim is not None:
            measureFreq(path=path, xlim = xlim)
        else:
            measureFreq(path=path)
    new_paths = []
    for path in paths:
        path = os.path.dirname(path)
        if path in new_paths:
            pass
        else:
            new_paths.append(path)
            for i in range(2):
                data_path = path + "/freqChannel{}.txt".format(i+1)
                if os.path.exists(data_path):
                    mean, u = calcUncerFromFile(data_path)
                    if u is None:
                        pass
                    else:
                        with open(data_path, "a")as file:
                            file.write("\n")
                            file.write("{} +- {}".format(mean, u))

def calcUncerFromFile(path):
    """
    Calculates the uncertainty and mean value.
    :param path: path to a .txt documents which contians the individual values
    :return: touple with mean and uncertainty information
    """
    with open(path, "r") as file:
        data = file.readlines()
    data_float = []
    for dp in data:
        data_float.append(float(dp.replace("\n", "")))
    sum = 0.0
    for dp in data_float:
        sum = sum + dp
    mean =sum / len(data_float)

    sigma = 0.0
    for dp in data_float:
        sigma = (dp - mean)**2
    if len(data_float) != 1 and len(data_float) != 0:
        sigma = sigma/(len(data_float)-1)
        sigma = np.sqrt(sigma)
    else:
        sigma = None

    return mean, sigma


def FFT(data, step, Tmin=None):
    """
    Calculates a FFT. Capable of downsampling if step < Tmin .
    :param data: amplitude data
    :param step: time step
    :param Tmin: Minimum time step used for down sampling
    :return: touple containing magnitute and frequency information
    """
    if Tmin is None: Tmin = 1e-7  # 1/fs
    if step < Tmin:
        maxlength = int(step / Tmin * len(data))
        length = len(data)
        data = downSample(data=data, maxlength=maxlength)
        step = int(length / len(data)) * step
    N = len(data)
    fft = np.fft.fft(data, N)
    data = None
    mag = abs(fft)*2/N
    fft = None
    freq = np.fft.fftfreq(N, step)
    return mag, freq

def binarySearch(data, key):
    """
    Classical binary search.
    :param data: array like
    :param key: search key
    :return: index closest to key or key
    """
    if len(data) == 1:
        return 0
    else:
        i = int(len(data)/2)-1
        if data[i] == key:
            return i
        elif data[i] > key:
            return binarySearch(data[0:i], key)
        elif data[i] < key:
            return i + binarySearch(data[i:-1], key)

def measureAmp(path, Tmax = None, xlim = None):
    """Evaluates the amplitude and width of the measured data. Has a Threshold implemented, and only amplitudes over that threshold are detected.
    :param Tmax : declares the maximum time between two samples. If this value is bigger than in the data, the data is downsampled.
    :param xlim : type touple: createsa rectangular over its limits"""
    data, status = extractData(path)

    for i, y in enumerate(data):
        if xlim is not None:
            y, t, step =RectWindow(dataElem=y, xlim=xlim, status=status)
        else:
            step = (status["HardwareXStop"] - status["HardwareXStart"]) / len(y)
        if Tmax is None: Tmax = 1e-7  # 1/fs
        if step < Tmax:
            maxlength = int(step / Tmax * len(y))
            length = len(y)
            y = downSample(data=y, maxlength=maxlength)
            step = int(length / maxlength) * step


        amplitudes = []

        if 1/step > 5e6:
            y = butter_highpass_filter(y, 3.8e+6, 1/step, 6)
        else:
            y = butter_lowpass_filter(y, 10e3, 1/step, 6)
        dydt = np.gradient(y, step)

        threshold = np.max(y)*0.8
        dy_threshold = 0.01*np.max(dydt)
        while len(amplitudes)<5:
            for j, dy in enumerate(dydt):
                if abs(dy) < dy_threshold and abs(y[j]) > threshold:
                    amplitudes.append(abs(y[j]))
            threshold = threshold - 0.01*threshold
            dy_threshold += 0.05*dy_threshold
        amplitude = np.sum(amplitudes) / len(amplitudes)


        file_path = os.path.dirname(path)
        with open(file_path + "/ampChannel{}.txt".format(i+1), "a") as file:
            file.write("{}\n".format(amplitude))

        width = []
        start = 0
        threshold = np.max(y)*0.9
        if i==0 and 1/step > 5e6:
            envelope = sig.hilbert(y)
            envelope = abs(envelope)
        else:
            envelope = y
        for j, yi in enumerate(envelope):
            if yi > threshold and start == 0:
                width.append(j)
                start = 1
                threshold = 0.1*np.max(y)
            elif yi < threshold and start == 1 and j > (width[0]+15):
                width.append(j)
                start = 0
                break
        if len(width) == 2:
            width = (width[1] - width[0])*step
        else:
            width = 0

        with open(file_path + "/widthChannel{}.txt".format(i + 1), "a") as file:
            file.write("{}\n".format(width))




def butter_highpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='high')
    return b, a

def butter_lowpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a

def butter_highpass_filter(data, highcut, fs, order=5):
    b, a = butter_highpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter(data, lowcut, fs, order=5):
    b, a = butter_lowpass(lowcut, fs, order)
    return lfilter(b, a, data)

def measureAmpAuto(path):
    """
    Measuers the amplitude of the measuerd data.
    :param path: given path and subfolders are searched for measured data
    :return: .txt file in same folder as measuerments containing the amplitude information and a mean and uncertainty value
    """
    paths = findPaths(path)
    for path in paths:  # Removes the stored data in .txt files if they exist
        path = os.path.dirname(path)
        for i in range(2):
            data = path + "/ampChannel{}.txt".format(i + 1)
            if os.path.exists(data):
                os.remove(data)
            data = path + "/widthChannel{}.txt".format(i + 1)
            if os.path.exists(data):
                os.remove(data)
    for path in paths:
        measureAmp(path)
    new_paths = []
    for path in paths:
        path = os.path.dirname(path)
        if path in new_paths:
            pass
        else:
            new_paths.append(path)
            for i in range(2):
                data_path_amp = path + "/ampChannel{}.txt".format(i + 1)
                data_path_width = path + "/widthChannel{}.txt".format(i + 1)
                if os.path.exists(data_path_amp):
                    mean, u = calcUncerFromFile(data_path_amp)
                    if u is None:
                        pass
                    else:
                        with open(data_path_amp, "a")as file:
                            file.write("\n")
                            file.write("{} +- {}".format(mean, u))
                if os.path.exists(data_path_width):
                    mean, u = calcUncerFromFile(data_path_width)
                    if u is None:
                        pass
                    else:
                        with open(data_path_width, "a")as file:
                            file.write("\n")
                            file.write("{} +- {}".format(mean, u))


def readValuesFromTxtAuto(path):
    """
    Reads all information from .txt files which were stored by the measuerFreqAuto or measureAmpAuto functions.
    :param path: given path and subfolders are searched for .txt files
    :return: A tree like structure containing all information from .txt file in location from path or subfolders
    """
    paths = findPaths(path=path, search_key=".txt")
    return_dict = {}
    for path in paths:
        data = readValuesFromTxt(path)
        path = path.split("/")
        path = path[1:]
        tmp = {path[-1] : data}
        for i in reversed(path[0:-1]):
            tmp = {i : tmp}
        return_dict = merge_dicts(tmp, return_dict)
    return return_dict

def merge_dicts(dict1, dict2):
    """
    Merges two dicts
    :param dict1:
    :param dict2:
    :return: The merged dictionary
    """
    if isinstance(dict1, type({})) and isinstance(dict2,  type({})):
        for key1 in dict1.keys():
            if key1 in dict2.keys():
                dict2[key1] = merge_dicts(dict1[key1], dict2[key1])
            else:
                dict2[key1] = dict1[key1]
        return dict2
    elif not isinstance(dict1, type({})) or not isinstance(dict2, type({})):
        return [dict1, dict2]








def readValuesFromTxt(path):
    """
    Reads data from .txt files which was evaluated by measureFreqAuto or measureAmpAuto functions.
    :param path: Path of the .txt file"
    :return: touple element containing the mean value and uncertainty value
    """
    if os.path.exists(path):
        with open(path, "r") as file:
            data = file.readlines()
            if len(data) == 0:
                raise AssertionError("There is no data contained in given path: {}".format(path))
        for i, point in enumerate(data):
            data[i] = point.replace("\n", "")
        data = data[-1]
        data = data.split("+-")
        for i, d in enumerate(data):
            data[i] = float(d)
        return data
    else:
        raise AssertionError("The file described by the path does not exist.")

def evaluate(path):
    """Extracts frequency and amplitude information of measurement data and returns it in a tree like structure.
    :param path = path to folder in which/ or in which subfolders, measuerments are stored"""
    measureTimingAuto(path=path)
    measureFreqAuto(path=path)
    measureAmpAuto(path=path)
    return readValuesFromTxtAuto(path=path)

def plotMeasurementsWithUncertainty(data, lastKey=""):
    """Reads information from the data and plots it. Plots are stored in "./plots" relative to the Plotter.py file.
    :param data = Output of readValuesFromTxt(), Tree like structure containing the data
    :return: plots in ./plots folder relative to Plotter.py"""
    if isinstance(data, type({})):
        flarmor = 4.3576
        ampMust = []
        teList = []
        rf90durationList =[]
        teFreqChannel1 = []
        durFreqChannel1 = []
        ampChannel1 = []
        ampChannel2 = []
        ampCurrChannel1 = []
        ampCurrChannel2 = []
        widthChannel1dur = []
        widthChannel1amp = []
        widthChannel1Curr = []
        widthChannel2Curr = []
        dBList = []
        amplificationList = []
        X = []
        Y = []
        for key in data.keys():
            if "mplitude" in key:
                if "Amplitude" in key:
                    amp = key.replace("Amplitude", "")
                else:
                    amp = key.replace("amplitude", "")
                if "mV" in amp:
                    amp = float(amp.replace("mV", ""))
                    tmp = data[key]["ampChannel1.txt"]
                    for i, t in enumerate(tmp):
                        tmp[i] = t*1000
                    if lastKey == "23032021PulseqRP":
                        tmp[0] = (tmp[0] - amp)
                    else:
                        tmp[0] = (tmp[0] - amp)*10**(30/20) #Measured new measurements with 30dB attenuator before oszilloscope
                        amp = amp * 10**(30/20)
                    tmp[0] = tmp[0]/amp*100
                    tmp[1] = tmp[1]/amp*100
                    ampChannel1.append(tmp)
                    amp = amp/1000
                    tmp = data[key]["widthChannel1.txt"]
                    tmp[0] = tmp[0]*1e6 - 40 #Standart width is 40 us for RF signals
                    tmp[1] = tmp[1]*1e6
                    widthChannel1amp.append(tmp)
                elif "A" in amp:
                    amp = float(amp.replace("A", ""))

                    Ch = ["ampChannel1.txt", "ampChannel2.txt"]
                    tpl = []
                    for c in Ch:
                        try:
                            tmp = data[key][c]
                        except:
                            continue
                        for i, t in enumerate(tmp):
                            tmp[i] = t*10000
                        tmp[0] = (tmp[0]/1000 - amp)/amp*100
                        tmp[1] = tmp[1]/1000/amp*100
                        tpl.append(tmp)
                    try:
                        ampCurrChannel1.append(tpl[0])
                        ampCurrChannel2.append(tpl[1])
                    except:
                        pass

                    Ch = ["widthChannel1.txt", "widthChannel2.txt"]
                    tpl = []
                    for c in Ch:
                        try:
                            tmp = data[key][c]
                        except:
                            continue
                        tmp[0] = tmp[0] * 1e6 - 4000  # Standart width is 4ms for gradients
                        tmp[1] = tmp[1] * 1e6
                        tpl.append(tmp)
                    try:
                        widthChannel1Curr.append(tpl[0])
                        widthChannel2Curr.append(tpl[1])
                    except:
                        pass
                ampMust.append(amp)

            elif "te" in key:
                te = key.replace("te", "")
                teList.append(float(te.replace("ms", "")))
                tmp = data[key]["freqChannel1.txt"]
                for i, t in enumerate(tmp):
                    if i == 0:
                        tmp[i] = t*1e-3 - flarmor*1e3
                    else:
                        tmp[i] = t*1e-3
                teFreqChannel1.append(tmp)
            elif "rf90duration" in key:
                rf90duration = key.replace("rf90duration", "")
                dur = float(rf90duration.replace("us", ""))
                rf90durationList.append(dur)
                tmp = data[key]["freqChannel1.txt"]
                for i, t in enumerate(tmp):
                    if i == 0:
                        tmp[i] = t*1e-3 - flarmor*1e3
                    else:
                        tmp[i] = t*1e-3
                durFreqChannel1.append(tmp)
                tmp = data[key]["widthChannel1.txt"]
                tmp[0] = tmp[0]*1e6 - dur
                tmp[1] = tmp[1]*1e6
                widthChannel1dur.append(tmp)
            elif "dB" in key:
                if "27052021RFAmplifier" == lastKey:
                    X.append(float(data[key]["ampChannel2.txt"][0])*1000)
                    Y.append(float(data[key]["ampChannel1.txt"][0])*1000)
                else:
                    dB = key.replace("dB", "")
                    dB = 400 * 10 ** (-float(dB) / 20)
                    dBList.append(dB)
                    tmp = data[key]["ampChannel1.txt"]
                    tmp[1] = 20 * np.log10(1000 * (tmp[0] + tmp[1]) / dB)
                    tmp[0] = 20 * np.log10(1000 * tmp[0] / dB)
                    tmp[1] = abs(tmp[1] - tmp[0])
                    amplificationList.append(tmp)
            else:
                plotMeasurementsWithUncertainty(data=data[key], lastKey=key)
        plotWithUncertainties(ampMust, ampChannel1, lastKey + "Amplitude")
        printUncer(ampChannel1, lastKey + "Amplitude")
        plotWithUncertainties(teList, teFreqChannel1, lastKey + "Te")
        printUncer(teFreqChannel1, lastKey +"Te")
        plotWithUncertainties(rf90durationList, durFreqChannel1, lastKey + "Rf90Duration")
        printUncer(durFreqChannel1, lastKey + "Rf90Duration")
        plotWithUncertainties(rf90durationList, widthChannel1dur, lastKey + "Rf90DurationWidth")
        printUncer(widthChannel1dur, lastKey + "Rf90DurationWidth")
        plotWithUncertainties(ampMust, widthChannel1amp, lastKey + "AmplitudeWidth")
        printUncer(widthChannel1amp, lastKey + "AmplitudeWidth")
        plotWithUncertainties(dBList, amplificationList, lastKey + "amplification")
        printUncer(amplificationList, lastKey + "amplification")
        plotWithUncertainties(ampMust, widthChannel1Curr, lastKey + "WidthGradientChannel1")
        printUncer(widthChannel1Curr, lastKey + "WidthGradientChannel1")
        plotWithUncertainties(ampMust, widthChannel2Curr, lastKey + "WidthGraidentChannel2")
        printUncer(widthChannel2Curr, lastKey + "WidthGraidentChannel2")
        plotWithUncertainties(ampMust, ampCurrChannel1, lastKey + "AmplitudeGradientChannel1")
        printUncer(ampCurrChannel1, lastKey + "AmplitudeGradientChannel1")
        plotWithUncertainties(ampMust, ampCurrChannel2, lastKey + "AmplitudeGradientChanel2")
        ampCurrChannel2, lastKey + "AmplitudeGradientChanel2"
        if len(X)!=0 and len(Y)!=0:
            X = bubblesort(X)
            Y = bubblesort(Y)
            x = np.linspace(X[0], X[-1], 100)
            y = 1/2*x
            plot([x,X], [y,Y], "RF Amplifier Linearity", xlabel="Input voltagein mV", ylabel="Output voltage in V", save=True, labelarray=["ideal", "measured"])
    else:
        pass

def printUncer(data, title):
    if len(data) > 1:
        mean = 0.0
        for dp in data:
            mean += dp[0]
        mean /= len(data)

        sigma = 0.0
        for dp in data:
            sigma = (dp[0] - mean)**2
        sigma = sigma/(len(data)-1)
        sigma = np.sqrt(sigma)
        print("Measurement with title " + title + " results in mean {} and sigma {}".format(mean, sigma))

def bubblesort(data):
    for i in range(len(data)):
        for k in range(len(data)-(i+1)):
            if data[k] > data[k+1]:
                tmp = data[k]
                data[k] = data[k+1]
                data[k+1] = tmp
    return data

def plotWithUncertainties(X, Y, title, save=True):
    """Plots the data in X and Y with errorbars. Each value of Y has two entires. The fist is the measured value. The second the uncertainty.
    X = data on x axis
    Y = data on y axis with uncertainty
    title = used for saving the figure
    save = if True saves the figure, if False shows the figure"""
    if X == [] or Y == []:
        pass
    else:
        plt.figure()
        if "Amplitude" in title:
            plt.xlabel("Amplitude set in V")
            plt.ylabel("(Delta V)/V_set in %")
        elif "Te" in title:
            plt.xlabel("Te in ms")
            plt.ylabel("Delta f in kHz")
        elif "Rf90Duration" in title:
            plt.xlabel("rf90duration in us")
            plt.ylabel("Delta f in kHz")
        elif "amplification" in title:
            plt.xlabel("Input amplitude in mV")
            plt.ylabel("Amplification in dB")
        if "Gradient" in title:
            plt.xlabel("Amplitude set in A")
            plt.ylabel("(Delta I)/I_set in %")
        if "Width" in title:
            plt.ylabel("Delta t in us")

        mean = 0.0
        for y in Y:
            mean += y[0]
        mean = mean/len(Y)
        for x, y in zip(X, Y):
            plt.errorbar(x, y[0], yerr=y[1], fmt=".k")
        plt.axhline(mean, color="r")
        plt.grid()
        plt.tight_layout()
        if save == True:
            savetitle = "./plots/" + title.replace(" ", "_") + ".pdf"
            plt.savefig(savetitle)
            plt.close("all")
        else:
            plt.show()

def measureTiming(path):
    """
    Extracts the time difference of the first and the second pulse of each recorded channel.
    :param: Path to folder containing data
    :return: timingChannel*.txt file in same file as data.
    """
    if isinstance(path, str):
        data, status = extractData(path)
        for i, d in enumerate(data):
            step = (status["HardwareXStop"] - status["HardwareXStart"]) / len(d)
            Tmin = 1e-7
            if step < Tmin:
                length = len(d)
                d = downSample(d, int(step/Tmin*len(d)))
                step = length/len(d) * step
            threshold = np.max(d)*0.6

            first = 0.0
            second = 0.0
            idx_first = 0
            for idx, point in enumerate(d):
                if point > threshold and first == 0.0:
                    first = idx*step
                    idx_first = idx
                elif point > threshold and (idx-idx_first)*step > 1e-4:
                    second = idx*step
                    break
                elif point > threshold:
                    idx_first = idx

            timing = second - first

            file_path = os.path.dirname(path)
            with open(file_path + "/timingChannel{}.txt".format(i + 1), "a") as file:
                file.write("{}\n".format(timing))
    else:
        raise ValueError("Wrong input argument!")


def measureTimingAuto(path):
    """
    Extracts the time difference of the first and the second pulse of each recorded channel.
    Does this procedure for every data contained in folder given by path and its subfolders.
    :param path: Path to data
    :return: timingChannel*.txt next to data
    """
    paths = findPaths(path)
    for path in paths:  # Removes the stored data in .txt files if they exist
        path = os.path.dirname(path)
        for i in range(2):
            data = path + "/timingChannel{}.txt".format(i + 1)
            if os.path.exists(data):
                os.remove(data)
    for path in paths:
        measureTiming(path)
    new_paths = []
    for path in paths:
        path = os.path.dirname(path)
        if path in new_paths:
            pass
        else:
            new_paths.append(path)
            for i in range(2):
                data_path = path + "/timingChannel{}.txt".format(i + 1)
                if os.path.exists(data_path):
                    mean, u = calcUncerFromFile(data_path)
                    if u is None:
                        pass
                    else:
                        with open(data_path, "a")as file:
                            file.write("\n")
                            file.write("{} +- {}".format(mean, u))




















if __name__ == '__main__':
    #TODO titledict only works if titles are keys are unique for every path
    data = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])*10/13000

    titledict = {}

    curr_path = os.path.dirname(__file__)   #returns the current path of the python skript
    curr_path = os.path.dirname(curr_path) + "/BA"  # ".."
    #curr_path = curr_path + "/22042021TRSwitchCoilToRx"
    #Autoplot(curr_path, titledict = titledict)
    #measureFreqAuto(path=curr_path)
    #measureAmpAuto(path=curr_path)
    #measureTimingAuto(curr_path)
    data = readValuesFromTxtAuto(path=curr_path)
    plotMeasurementsWithUncertainty(data=data)


    save = False

    freqLim = (0e6, 1e6)
    path = "/home/ole/StorageServer/Work/BachelorThesis/RedPitaya16BitUndMessaufbau/Messergebnisse/NeueMessungenAbRFAmp/27052021TRSwitch/RFAmplifierGaPulse/RefCurve_2021-05-27_0_143038"
    xlim = None # (0.0, 0.00003)
    
    #Plotter(paths=[path], title ="RF Amplifier With Gate Pulse", save=save, xlim=xlim, multiply=[10**(30/20), 1], labelarray=["RF Pulse", "Gate Pulse"])
    #FFTPlot(path=path, xlim=xlim, save=save, freqLim=freqLim)

    #path = "/home/ole/StorageServer/Work/BachelorThesis/RedPitaya16BitUndMessaufbau/Messergebnisse/NeueMessungenAbRFAmp/27052021TRSwitch/amplitude251.95mV/RefCurve_2021-05-27_0_140507"
    #Plotter(paths=[path], title ="TR Switch Output", save=save, multiply=[10**(30/20)], labelarray=["RF Pulse"])

    path = "/home/ole/StorageServer/Work/BachelorThesis/RedPitaya16BitUndMessaufbau/Messergebnisse/BA/22042021RPOCRABased/RefCurve_2021-04-22_0_143403"
    #Plotter(paths=[path], title="RP Ocra based Output", save=save, labelarray=["RF Pulse"])

    path ="/home/ole/StorageServer/Work/BachelorThesis/RedPitaya16BitUndMessaufbau/Messergebnisse/BA/23032021PulseqRP/te12ms/RefCurve_2021-03-23_0_134215"
    #Plotter(paths=[path], title="RP PulSeq based Output", save=save, labelarray=["RF Pulse"])

    path = "/home/ole/StorageServer/Work/BachelorThesis/RedPitaya16BitUndMessaufbau/Messergebnisse/BA/19052021GPAFHDO/Channel1/te12ms/RefCurve_2021-05-19_0_152231"
    #Plotter(paths=[path], title="Scanner Output Timing TE 12", save=save, labelarray=["RF Pulse", "Gradient Pulse"])

    path ="/home/ole/StorageServer/Work/BachelorThesis/RedPitaya16BitUndMessaufbau/Messergebnisse/BA/19052021GPAFHDO/Channel1/te6ms/RefCurve_2021-05-19_0_155239"
    #Plotter(paths=[path], title="Scanner Output Timing TE 6", save=save, labelarray=["RF Pulse", "Gradient Pulse"])

    freqLim = (0, 6e+6)
    path ="/home/ole/StorageServer/Work/BachelorThesis/RedPitaya16BitUndMessaufbau/Messergebnisse/BA/NeueMessungenAbRFAmp/27052021TRSwitch/RFAmplifierGaPulse/RefCurve_2021-05-27_0_143038"
    #FFTPlot(path=path, title="FFT spike gate pulse", xlim=(-0.02e-3,0.04e-3), save=save, freqLim=freqLim)






