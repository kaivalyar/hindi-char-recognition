import os
from keras.models import load_model

base_list = [2310, 2311, 2312, 2313, 2315, 2319, 2325, 2327, 2328, 2330, 2331,
       2332, 2334, 2335, 2336, 2337, 2338, 2340, 2341, 2342, 2343, 2344,
       2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2357, 2358,
       2359, 2360, 2361, 2384, 2392, 2399, 2404, 2405, 2406, 2407, 2408,
       2409, 2410, 2411, 2412, 2413, 2414, 2415, 2424, 2429]
matra_list = [0, 2363, 2366, 2367, 2368, 2369, 2370, 2372, 2375, 2376, 2379,
       2380, 2382, 2384, 2387, 2390]
dot_list = [0, 2306, 2364]

base_model = load_model('models/base_model3k.model')
matra_model = load_model('models/matra_model3k.model')
dot_model = load_model('models/dot_model3k.model')

path = '../train_images_original'
tests = [path+'/'+filename for filename in os.listdir(path) if filename.endswith(".png")]

[
'tests/page0_5_8_2346.png',
'tests/page0_5_9_2340_2368.png',
'tests/page0_5_10_2352.png',
'tests/page0_5_12_2357_2366.png',
'tests/page0_6_6_2340_2366.png',
'tests/page0_6_7_2325.png',
'tests/page0_6_8_2352.png',
'tests/page0_6_9_2340_2366.png',
'tests/page0_7_5_2361_2366_2417.png',
'tests/page0_7_7_2330_2367.png',
'tests/page0_7_8_2361_2376.png',
'tests/page0_7_9_2424.png',
'tests/page0_8_8_2330_2375_2379.png',
'tests/page0_8_9_2346.png',
'tests/page0_8_10_2312.png',
'tests/page0_8_14_2352.png',
'tests/page0_9_6_2361.png',
'tests/page0_9_8_2312.png',
'tests/page0_9_9_2340.png',
'tests/page0_9_10_2344.png',
]
