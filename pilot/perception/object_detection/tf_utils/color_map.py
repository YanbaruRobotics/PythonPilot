import numpy as np

STANDARD_COLORS_PIL = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

''' PIL RGB colors to OpenCV BGR
for color in STANDARD_COLORS:
    rgb = ImageColor.getrgb(color)
    bgr = (rgb[2],rgb[1],rgb[0])
    print("  {}, # {}".format(bgr,color))
'''

"""
BGR COLORS
"""
STANDARD_COLORS = [
    (0, 0, 0), #Black
    (60, 180, 75), #Green
    (230, 25, 75), #Red
    (255, 225, 25), #Yellow
    (0, 130, 200), #Blue
    (245, 130, 48), #Orange
    (145, 30, 180), #Purple
    (70, 240, 240), #Cyan
    (0, 128, 128), #Teal
    (210, 245, 60), #Lime
    (250, 190, 190), #Pink
    (240, 50, 230), #Magenta
    (230, 190, 255), #Lavender
    (170, 110, 40), #Brown
    (255, 250, 200), #Beige
    (0, 252, 124), # LawnGreen
    (128, 0, 0), #Maroon
    (170, 255, 195), #Mint
    (128, 128, 0), #Olive
    (255, 215, 180), #Coral
    (0, 0, 128), #Navy
    (255, 255, 255), #White
    (0, 255, 127), # Chartreuse
    (255, 255, 0), # Aqua
    (212, 255, 127), # Aquamarine
    (255, 255, 240), # Azure
    (220, 245, 245), # Beige
    (196, 228, 255), # Bisque
    (205, 235, 255), # BlanchedAlmond
    (226, 43, 138), # BlueViolet
    (135, 184, 222), # BurlyWood
    (160, 158, 95), # CadetBlue
    (215, 235, 250), # AntiqueWhite
    (30, 105, 210), # Chocolate
    (80, 127, 255), # Coral
    (237, 149, 100), # CornflowerBlue
    (220, 248, 255), # Cornsilk
    (60, 20, 220), # Crimson
    (255, 255, 0), # Cyan
    (139, 139, 0), # DarkCyan
    (11, 134, 184), # DarkGoldenRod
    (169, 169, 169), # DarkGrey
    (107, 183, 189), # DarkKhaki
    (0, 140, 255), # DarkOrange
    (204, 50, 153), # DarkOrchid
    (122, 150, 233), # DarkSalmon
    (143, 188, 143), # DarkSeaGreen
    (209, 206, 0), # DarkTurquoise
    (211, 0, 148), # DarkViolet
    (147, 20, 255), # DeepPink
    (255, 191, 0), # DeepSkyBlue
    (255, 144, 30), # DodgerBlue
    (34, 34, 178), # FireBrick
    (240, 250, 255), # FloralWhite
    (34, 139, 34), # ForestGreen
    (255, 0, 255), # Fuchsia
    (220, 220, 220), # Gainsboro
    (255, 248, 248), # GhostWhite
    (0, 215, 255), # Gold
    (32, 165, 218), # GoldenRod
    (114, 128, 250), # Salmon
    (140, 180, 210), # Tan
    (240, 255, 240), # HoneyDew
    (180, 105, 255), # HotPink
    (92, 92, 205), # IndianRed
    (240, 255, 255), # Ivory
    (140, 230, 240), # Khaki
    (250, 230, 230), # Lavender
    (245, 240, 255), # LavenderBlush
    (255, 248, 240), # AliceBlue
    (205, 250, 255), # LemonChiffon
    (230, 216, 173), # LightBlue
    (128, 128, 240), # LightCoral
    (255, 255, 224), # LightCyan
    (210, 250, 250), # LightGoldenRodYellow
    (211, 211, 211), # LightGray
    (211, 211, 211), # LightGrey
    (144, 238, 144), # LightGreen
    (193, 182, 255), # LightPink
    (122, 160, 255), # LightSalmon
    (170, 178, 32), # LightSeaGreen
    (250, 206, 135), # LightSkyBlue
    (153, 136, 119), # LightSlateGray
    (153, 136, 119), # LightSlateGrey
    (222, 196, 176), # LightSteelBlue
    (224, 255, 255), # LightYellow
    (0, 255, 0), # Lime
    (50, 205, 50), # LimeGreen
    (230, 240, 250), # Linen
    (255, 0, 255), # Magenta
    (170, 205, 102), # MediumAquaMarine
    (211, 85, 186), # MediumOrchid
    (219, 112, 147), # MediumPurple
    (113, 179, 60), # MediumSeaGreen
    (238, 104, 123), # MediumSlateBlue
    (154, 250, 0), # MediumSpringGreen
    (204, 209, 72), # MediumTurquoise
    (133, 21, 199), # MediumVioletRed
    (250, 255, 245), # MintCream
    (225, 228, 255), # MistyRose
    (181, 228, 255), # Moccasin
    (173, 222, 255), # NavajoWhite
    (230, 245, 253), # OldLace
    (0, 128, 128), # Olive
    (35, 142, 107), # OliveDrab
    (0, 165, 255), # Orange
    (0, 69, 255), # OrangeRed
    (214, 112, 218), # Orchid
    (170, 232, 238), # PaleGoldenRod
    (152, 251, 152), # PaleGreen
    (238, 238, 175), # PaleTurquoise
    (147, 112, 219), # PaleVioletRed
    (213, 239, 255), # PapayaWhip
    (185, 218, 255), # PeachPuff
    (63, 133, 205), # Peru
    (203, 192, 255), # Pink
    (221, 160, 221), # Plum
    (230, 224, 176), # PowderBlue
    (128, 0, 128), # Purple
    (0, 0, 255), # Red
    (143, 143, 188), # RosyBrown
    (225, 105, 65), # RoyalBlue
    (19, 69, 139), # SaddleBrown
    (0, 128, 0), # Green
    (96, 164, 244), # SandyBrown
    (87, 139, 46), # SeaGreen
    (238, 245, 255), # SeaShell
    (45, 82, 160), # Sienna
    (192, 192, 192), # Silver
    (235, 206, 135), # SkyBlue
    (205, 90, 106), # SlateBlue
    (144, 128, 112), # SlateGray
    (144, 128, 112), # SlateGrey
    (250, 250, 255), # Snow
    (127, 255, 0), # SpringGreen
    (180, 130, 70), # SteelBlue
    (47, 255, 173), # GreenYellow
    (128, 128, 0), # Teal
    (216, 191, 216), # Thistle
    (71, 99, 255), # Tomato
    (208, 224, 64), # Turquoise
    (238, 130, 238), # Violet
    (179, 222, 245), # Wheat
    (255, 255, 255), # White
    (245, 245, 245), # WhiteSmoke
    (0, 255, 255), # Yellow
    (50, 205, 154), # YellowGreen
]

STANDARD_COLORS_ARRAY = np.asarray(STANDARD_COLORS).astype(np.uint8)
