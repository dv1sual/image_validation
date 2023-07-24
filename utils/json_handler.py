import json

# All three color chart values
aces2065_1_color_chart_values = [
    {
        'name': 'Dark Skin',
        'x': 215,
        'y': 165,
        'rgb': (0.118, 0.088, 0.062)
    },
    {
        'name': 'Light Skin',
        'x': 515,
        'y': 165,
        'rgb': (0.401, 0.310, 0.234)
    },
    {
        'name': 'Blue Sky',
        'x': 815,
        'y': 165,
        'rgb': (0.180, 0.201, 0.311)
    },
    {
        'name': 'Foliage',
        'x': 1115,
        'y': 165,
        'rgb': (0.111, 0.135, 0.064)
    },
    {
        'name': 'Blue Flower',
        'x': 1415,
        'y': 165,
        'rgb': (0.258, 0.238, 0.405)
    },
    {
        'name': 'Bluish Green',
        'x': 1715,
        'y': 165,
        'rgb': (0.317, 0.468, 0.420)
    },
    {
        'name': 'Orange',
        'x': 215,
        'y': 408,
        'rgb': (0.410, 0.233, 0.062)
    },
    {
        'name': 'Purplish Blue',
        'x': 515,
        'y': 408,
        'rgb': (0.137, 0.130, 0.361)
    },
    {
        'name': 'Moderate Red',
        'x': 815,
        'y': 408,
        'rgb': (0.303, 0.131, 0.123)
    },
    {
        'name': 'Purple',
        'x': 1115,
        'y': 408,
        'rgb': (0.091, 0.058, 0.130)
    },
    {
        'name': 'Yellow Green',
        'x': 1415,
        'y': 408,
        'rgb': (0.355, 0.448, 0.110)
    },
    {
        'name': 'Orange Yellow',
        'x': 1715,
        'y': 408,
        'rgb': (0.490, 0.365, 0.075)
    },
    {
        'name': 'Blue',
        'x': 215,
        'y': 649,
        'rgb': (0.079, 0.071, 0.258)
    },
    {
        'name': 'Green',
        'x': 515,
        'y': 649,
        'rgb': (0.151, 0.255, 0.096)
    },
    {
        'name': 'Red',
        'x': 815,
        'y': 649,
        'rgb': (0.220, 0.070, 0.047)
    },
    {
        'name': 'Yellow',
        'x': 1115,
        'y': 649,
        'rgb': (0.595, 0.538, 0.089)
    },
    {
        'name': 'Magenta',
        'x': 1415,
        'y': 649,
        'rgb': (0.324, 0.151, 0.285)
    },
    {
        'name': 'Cyan',
        'x': 1715,
        'y': 649,
        'rgb': (0.149, 0.236, 0.374)
    },
    {
        'name': 'White 9.5 (.05 D)',
        'x': 215,
        'y': 900,
        'rgb': (0.910, 0.913, 0.897)
    },
    {
        'name': 'Neutral 8 (.23 D)',
        'x': 515,
        'y': 900,
        'rgb': (0.587, 0.591, 0.593)
    },
    {
        'name': 'Neutral 6.5 (.44 D)',
        'x': 815,
        'y': 900,
        'rgb': (0.361, 0.365, 0.367)
    },
    {
        'name': 'Neutral 5 (.70 D)',
        'x': 1115,
        'y': 900,
        'rgb': (0.191, 0.191, 0.193)
    },
    {
        'name': 'Neutral 3.5 (1.05 D)',
        'x': 1415,
        'y': 900,
        'rgb': (0.088, 0.089, 0.092)
    },
    {
        'name': 'Black 2 (1.5 D)',
        'x': 1715,
        'y': 900,
        'rgb': (0.031, 0.031, 0.033)
    }
]

itu_r_bt_2020_color_chart_values = [
    {
        'name': 'Dark Skin',
        'x': 215,
        'y': 165,
        'rgb': (0.138, 0.088, 0.061)
    },
    {
        'name': 'Light Skin',
        'x': 515,
        'y': 165,
        'rgb': (0.462, 0.311, 0.232)
    },
    {
        'name': 'Blue Sky',
        'x': 815,
        'y': 165,
        'rgb': (0.145, 0.192, 0.314)
    },
    {
        'name': 'Foliage',
        'x': 1115,
        'y': 165,
        'rgb': (0.114, 0.144, 0.062)
    },
    {
        'name': 'Blue Flower',
        'x': 1415,
        'y': 165,
        'rgb': (0.230, 0.220, 0.409)
    },
    {
        'name': 'Bluish Green',
        'x': 1715,
        'y': 165,
        'rgb': (0.254, 0.484, 0.418)
    },
    {
        'name': 'Orange',
        'x': 215,
        'y': 408,
        'rgb': (0.536, 0.236, 0.057)
    },
    {
        'name': 'Purplish Blue',
        'x': 515,
        'y': 408,
        'rgb': (0.090, 0.107, 0.368)
    },
    {
        'name': 'Moderate Red',
        'x': 815,
        'y': 408,
        'rgb': (0.389, 0.119, 0.124)
    },
    {
        'name': 'Purple',
        'x': 1115,
        'y': 408,
        'rgb': (0.090, 0.049, 0.132)
    },
    {
        'name': 'Yellow Green',
        'x': 1415,
        'y': 408,
        'rgb': (0.384, 0.490, 0.100)
    },
    {
        'name': 'Orange Yellow',
        'x': 1715,
        'y': 408,
        'rgb': (0.616, 0.384, 0.067)
    },
    {
        'name': 'Blue',
        'x': 215,
        'y': 649,
        'rgb': (0.041, 0.052, 0.263)
    },
    {
        'name': 'Green',
        'x': 515,
        'y': 649,
        'rgb': (0.136, 0.279, 0.092)
    },
    {
        'name': 'Red',
        'x': 815,
        'y': 649,
        'rgb': (0.298, 0.060, 0.047)
    },
    {
        'name': 'Yellow',
        'x': 1115,
        'y': 649,
        'rgb': (0.722, 0.579, 0.077)
    },
    {
        'name': 'Magenta',
        'x': 1415,
        'y': 649,
        'rgb': (0.379, 0.123, 0.290)
    },
    {
        'name': 'Cyan',
        'x': 1715,
        'y': 649,
        'rgb': (0.076, 0.230, 0.378)
    },
    {
        'name': 'White 9.5 (.05 D)',
        'x': 215,
        'y': 900,
        'rgb': (0.912, 0.915, 0.896)
    },
    {
        'name': 'Neutral 8 (.23 D)',
        'x': 515,
        'y': 900,
        'rgb': (0.584, 0.591, 0.593)
    },
    {
        'name': 'Neutral 6.5 (.44 D)',
        'x': 815,
        'y': 900,
        'rgb': (0.359, 0.365, 0.367)
    },
    {
        'name': 'Neutral 5 (.70 D)',
        'x': 1115,
        'y': 900,
        'rgb': (0.191, 0.192, 0.193)
    },
    {
        'name': 'Neutral 3.5 (1.05 D)',
        'x': 1415,
        'y': 900,
        'rgb': (0.087, 0.089, 0.092)
    },
    {
        'name': 'Black 2 (1.5 D)',
        'x': 1715,
        'y': 900,
        'rgb': (0.031, 0.031, 0.033)
    }
]


RGB_color_chart_values = [
    {
        'name': 'Dark Skin',
        'x': 215,
        'y': 165,
        'rgb': (115, 82, 68)
    },
    {
        'name': 'Light Skin',
        'x': 515,
        'y': 165,
        'rgb': (194, 150, 130)
    },
    {
        'name': 'Blue Sky',
        'x': 815,
        'y': 165,
        'rgb': (98, 122, 157)
    },
    {
        'name': 'Foliage',
        'x': 1115,
        'y': 165,
        'rgb': (87, 108, 67)
    },
    {
        'name': 'Blue Flower',
        'x': 1415,
        'y': 165,
        'rgb': (133, 128, 177)
    },
    {
        'name': 'Bluish Green',
        'x': 1715,
        'y': 165,
        'rgb': (103, 189, 170)
    },
    {
        'name': 'Orange',
        'x': 215,
        'y': 408,
        'rgb': (214, 126, 44)
    },
    {
        'name': 'Purplish Blue',
        'x': 515,
        'y': 408,
        'rgb': (80, 91, 166)
    },
    {
        'name': 'Moderate Red',
        'x': 815,
        'y': 408,
        'rgb': (193, 90, 99)
    },
    {
        'name': 'Purple',
        'x': 1115,
        'y': 408,
        'rgb': (94, 60, 108)
    },
    {
        'name': 'Yellow Green',
        'x': 1415,
        'y': 408,
        'rgb': (157, 188, 64)
    },
    {
        'name': 'Orange Yellow',
        'x': 1715,
        'y': 408,
        'rgb': (224, 163, 46)
    },
    {
        'name': 'Blue',
        'x': 215,
        'y': 649,
        'rgb': (56, 61, 150)
    },
    {
        'name': 'Green',
        'x': 515,
        'y': 649,
        'rgb': (70, 148, 73)
    },
    {
        'name': 'Red',
        'x': 815,
        'y': 649,
        'rgb': (175, 54, 60)
    },
    {
        'name': 'Yellow',
        'x': 1115,
        'y': 649,
        'rgb': (231, 199, 31)
    },
    {
        'name': 'Magenta',
        'x': 1415,
        'y': 649,
        'rgb': (187, 86, 149)
    },
    {
        'name': 'Cyan',
        'x': 1715,
        'y': 649,
        'rgb': (8, 133, 161)
    },
    {
        'name': 'White 9.5 (.05 D)',
        'x': 215,
        'y': 900,
        'rgb': (243, 243, 243)
    },
    {
        'name': 'Neutral 8 (.23 D)',
        'x': 515,
        'y': 900,
        'rgb': (200, 200, 200)
    },
    {
        'name': 'Neutral 6.5 (.44 D)',
        'x': 815,
        'y': 900,
        'rgb': (160, 160, 160)
    },
    {
        'name': 'Neutral 5 (.70 D)',
        'x': 1115,
        'y': 900,
        'rgb': (122, 122, 121)
    },
    {
        'name': 'Neutral 3.5 (1.05 D)',
        'x': 1415,
        'y': 900,
        'rgb': (85, 85, 85)
    },
    {
        'name': 'Black 2 (1.5 D)',
        'x': 1715,
        'y': 900,
        'rgb': (52, 52, 52)
    }
]

# Combine all color chart values into one dictionary
color_chart_values = {
    "aces2065_1_color_chart_values": aces2065_1_color_chart_values,
    "itu_r_bt_2020_color_chart_values": itu_r_bt_2020_color_chart_values,
    "RGB_color_chart_values": RGB_color_chart_values
}

# Write the color chart values to a JSON file
with open('/color_charts_values/color_chart_values.json', 'w') as file:
    json.dump(color_chart_values, file)