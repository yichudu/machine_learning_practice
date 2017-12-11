#from keras.models import Model


def generate_arrays():
    X=[1,2,3]
    y=sum(X)
    num=0
    while True:
        if num>100:
            break
        X=[x+1 for x in X]
        y = sum(X)
        yield (X,y)
        num+=1

for X,y in generate_arrays():
        print(X,y)

#
# model=Model()
# model.fit_generator(generator=generate_arrays)