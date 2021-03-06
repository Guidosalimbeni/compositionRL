# direct inputs
# source to this solution and code:
# http://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game
# http://www.gamespp.com/directx/directInputKeyboardScanCodes.html

import ctypes

SendInput = ctypes.windll.user32.SendInput


W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

NP_1 = 0x4F
NP_2 = 0x50
NP_3 = 0x51
NP_4 = 0x4B
NP_6 = 0x4D
NP_8 = 0x48

J = 0x24
K = 0x25
L = 0x26
I = 0x17


# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    ReleaseKey(I)
    ReleaseKey(L)
    ReleaseKey(K)
    ReleaseKey(J)


def left():
    ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    ReleaseKey(I)
    ReleaseKey(L)
    ReleaseKey(K)
    ReleaseKey(J)

def right():

    ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(I)
    ReleaseKey(L)
    ReleaseKey(K)
    ReleaseKey(J)

def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(I)
    ReleaseKey(L)
    ReleaseKey(K)
    ReleaseKey(J)
    
def select1():
    PressKey(J)
    ReleaseKey(K)
    ReleaseKey(I)
    ReleaseKey(L)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)
    
def select2():
    ReleaseKey(I)
    PressKey(K)
    ReleaseKey(L)
    ReleaseKey(J)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)
    
def select3():
    ReleaseKey(I)
    ReleaseKey(K)
    PressKey(L)
    ReleaseKey(J)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)

def select4():
    ReleaseKey(J)
    ReleaseKey(K)
    ReleaseKey(L)
    PressKey(I)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)
    
# =============================================================================
# 
# def forward_left():
#     PressKey(W)
#     PressKey(A)
#     ReleaseKey(D)
#     ReleaseKey(S)
# 
# 
# def forward_right():
#     PressKey(W)
#     PressKey(D)
#     ReleaseKey(A)
#     ReleaseKey(S)
# 
# 
# def reverse_left():
#     PressKey(S)
#     PressKey(A)
#     ReleaseKey(W)
#     ReleaseKey(D)
# 
# 
# def reverse_right():
#     PressKey(S)
#     PressKey(D)
#     ReleaseKey(W)
#     ReleaseKey(A)
# =============================================================================
