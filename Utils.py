import fnmatch,os

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result
file=find('*.csv', 'test2')
prediction_key = ""
a=["empty","岩羊","未知","耗牛","羚牛","血雉","鼠兔"]
with open(file[0], "rt",encoding='GB2312') as f:
    reader1 = csv.reader(f, delimiter = ',')
    for row in reader1:
        for elements in a:
            if row[10] == elements :
                predictor = prediction_endpoint.PredictionEndpoint(prediction_key)    
            with open("images/Test/test_od_image.jpg", mode="rb") as test_data:
                results = predictor.predict_image(project.id, test_data, iteration.id)
            for prediction in results.predictions:
                print ("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100), prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height)
