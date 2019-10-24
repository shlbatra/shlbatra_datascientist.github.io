

# Starting with python studies

![output_13_3.png](attachment:output_13_3.png)

### Essential CheatSheets: https://startupsventurecapital.com/essential-cheat-sheets-for-machine-learning-and-deep-learning-researchers-efb6a8ebd2e5

# Index for Contents:
### Python Basics
[1.1-Python-Basics-Functions](1.1-Python-Basics-Functions.ipynb) <br/>
[1.2 Python-Basics-Control-Directoy](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>

### Data Types
[1.1-Python-Basics-Functions](1.1-Python-Basics-Functions.ipynb) <br/>
[1.2 Python-Basics-Control-Directoy](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>

### Data Visualization
[1.1-Python-Basics-Functions](1.1-Python-Basics-Functions.ipynb) <br/>
[1.2 Python-Basics-Control-Directoy](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>
[1.3 Python-Basics-Functions](1.1-Python-Basics---Functions.ipynb) <br/>

## importing important libraries


```python
import numpy as np
import pandas as pd
```


```python
#setting precision to 2
%precision 2
```




    '%.2f'



## DIRECTORY CONTROL


```python
import os
print(os.getcwd()) #printint current directory
os.listdir()
#using listdir on another folder that isn't the current directory
dir_to_list = "C:/Users/mateus_silva1/Desktop/Files, Texts, and Others"
os.listdir(dir_to_list)
#to create a new directory uses os.mkdir("folder_name")
#to rename a directory: os.rename('from','to')
#to remove uses os.remove
#remove nom empty directory: import shutil, shutil.rmtree('test_folder')
#TO change WORK Directory: os.chdir('path')
```

    C:\Users\mateus_silva1\Python_Jupyter_Studies
    




    ['12 Use Cases for Parameters.twbx',
     'AMERICAN ENGLISH FILE 2',
     'BMS CUBE.pbix',
     'data science dell certifications',
     'data-science-cheatsheet.pdf',
     'documents and appointsments',
     'Dynamic FY dimDate Table.pbix',
     'english_teaching_certif',
     'GANT SRO.png',
     'Icons',
     'images prints',
     'journal fr.txt',
     'kpis physical.png',
     'Marty_GS_Guard_ListGlobal Contact List.xlsx',
     'Mateus Leao Talent Profile.pdf',
     'Mateus S Leao  CV.docx',
     'Mateus S Leao - Dell - CV.pdf',
     'mov',
     'ODM Listings v1 03282019.xlsx',
     'physical security snippet.png',
     'Portables',
     'SecurityIncidents_July20192019-08-08 11-34-56.csv',
     'SecurityIncidents_July20192019-08-08 11-34-56.xlsx',
     'SRO-P Investigations Procedure - Investigations List - 6May19.xlsx',
     'SRO_Facilities_Data.xlsx',
     'Subs',
     'tableau date exercise.xlsx',
     'the osb',
     'txt fc',
     'User Stories Abraham and Saed.docx',
     'visto',
     'WEF_Global_Risks_Report_2019.pdf',
     'WPV Case Management Process Flow.vsdx']



### Directory Control and Reading data with Pandas


```python
import os
import pandas as pd
current_dir = os.getcwd()
print ("current directory is: ", current_dir)
dir_to_list = "C:/Users/mateus_silva1/Documents/Data Science/Python Scripts"
os.listdir(dir_to_list)
os.chdir("C:/Users/mateus_silva1/Documents/Data Science/Python Scripts") #changing work directory
df = pd.read_csv('mtcars.csv', index_col=0)
df.iloc[0:5,0:3]
#READ EXCEL FILE df = pd.read_excel('File.xlsx', sheetname='Sheet1')
```

### OPENING FILES and WRITING and Reading and Closing


```python
###OPENING A FILE WITH PANDAS
columns = ['a','b','c']
df = pd.read_csv('file', names=columns, encoding ='latin-1')
#open file in current directory:
f = open("test.txt")
ff = open ("C:/...")
try:
f = open("test.txt", mode= 'r', enconding = "utf-8")
finally:
f.close() #closing file
```


      File "<ipython-input-37-ac5b8959d47f>", line 8
        f = open("test.txt", mode= 'r', enconding = "utf-8")
        ^
    IndentationError: expected an indented block
    


### Creating file (with open) and Reading it


```python
#create a file and read it
#to read file line by line 
with open ("test_write_python.txt", mode= 'w', encoding = 'utf-8') as f:
           f.write("my first line writing in python \n")
           f.write("writting a big paragraph is just about time and dedication, the more you write...")
f.close()

ff = open("test_write_python.txt", mode = 'r', encoding = 'utf-8')
for line in ff:
    print (line)
```

### Functions w/ try and except:


```python
#defining function and try and except (handling with exception errors)
def dividing_but_not_by_zero(x1,x2):
    try:
        result = x1/x2 #this variable is in the local scope
        print(result)
        #normal except is just except:
        except ZeroDivisionError:
        print("exception error by zero")
dividing_but_not_by_zero(2,2)

#Another simple function:
def number_to_square(x):
    result = x**2
    return result
square = number_to_square(5)
print (square)
```

    1.0
    

### Lambda Python Functions


```python
#simple multiplier
x = lambda a,b : a*b
print (x(5,5))

#lambda as a variable
potency_function = lambda num, pot: num**pot
potency_function(2,3)
print("pot result is ", potency_function)
#INSIDE ANOTHER FUNCTION (n is in the bigger function, a is in the lambda function)
def multiplier (n):
    return lambda x: x*n #RETURNING A FUNCTION 
mytripler = multiplier(3) #AND PUTTING IN A VARIABLE :o
mydoubler = multiplier(2) #and putting in a variable
print ("value from doubler is: ", mydoubler(10))
print ("value from tripler is: ", mytripler(10))

```

    25
    pot result is  <function <lambda> at 0x00000262462D11E0>
    value from doubler is:  20
    value from tripler is:  30
    

### Apply(lambda x: function(x)) - apply function to arrays for example


```python
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text

data['body_text_nostop'] = data['body_text_tokenized'].apply(lambda x: remove_stopwords(x))
```

#### Lambda and Map


```python
people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']

def split_title_and_name(person):
    return person.split()[0] + ' ' + person.split()[-1]

#option 1
for person in people:
    print((lambda x: x.split()[0] + ' ' + x.split()[-1])(person))

#option 2
print(list(map(lambda person: person.split()[0] + ' ' + person.split()[-1], people)))
```

    Dr. Brooks
    Dr. Collins-Thompson
    Dr. Vydiswaran
    Dr. Romero
    ['Dr. Brooks', 'Dr. Collins-Thompson', 'Dr. Vydiswaran', 'Dr. Romero']
    

#### Map Function: store result in a map iterable file


```python
def to_upper_case(s):
    return str(s).upper()
map_iterator_upper = map(to_upper_case, 'abc')
for item in map_iterator_upper:
    print (item)

list_numbers = [1, 2, 3, 4]
map_iterator = map(lambda x: x * 2, list_numbers)
for item in map_iterator:
    print (item)
```

    A
    B
    C
    2
    4
    6
    8
    

### JSON Files

### Iterators


```python
list_created = ([1,2,3,4,5])
iter_object = iter(list_created)
print(next(iter_object))
print(next(iter_object))
print(next(iter_object))
print(next(iter_object))
print(next(iter_object))
```

    1
    2
    3
    4
    5
    

### For Loops: Break, Continue, pass


```python
for i in range(100):
    print (i)
    if i == 10:
        break #break when I = 10

        
list_created = ([10000,2,-5,100,-2])
#below: when i is positive just print it and continue to next iteration, if not, turn to positive
for i in list_created:
    if i>0:
        print(i)
        continue
    i = i*-1
    print(i)
    
#pass do nothing, is provisory fix for having a function for example that has no body yet
for i in list_created:
    pass
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    10000
    2
    5
    100
    2
    

### For Loop with index position and format to print


```python
data = ['John', 'Doe', 'was', 'here']
for idx, val in enumerate(data): #ENUMERATE BRINGS US THE INDEX and also the value
    print('{}: {}'.format(idx, val))
```

    0: John
    1: Doe
    2: was
    3: here
    

## DATA TIME WORK

#### super basics of datetime from timestamp


```python
import datetime as dt
import time as tm
```


```python
tm.time()
dtnow = dt.datetime.fromtimestamp(tm.time())
dtnow
dtnow.year, dtnow.month,dtnow.hour
```




    (2019, 10, 0)




```python
delta = dt.timedelta(days = 100)
print("delta:", delta)
today = dt.date.today()
print ("today:", today)
print("today - delta:", today-delta)
print("today > delta?:", today > (today-delta)

```

    delta: 100 days, 0:00:00
    today: 2019-10-17
    today - delta: 2019-07-09
    today > delta?: True
    

# Data types (lists, dictionaries, tulpas, strings)

## Lists

#### List Methods

<img src="C:/Users/mateus_silva1/Python_Jupyter_Studies/images/list_methods.PNG">


```python
#loading image with code
from IPython.display import Image
Image(filename='C:/Users/mateus_silva1/Python_Jupyter_Studies/images/list_methods.PNG')
```




![png](output_41_0.png)



### creating Lists, append and extend (concatenate)


```python
#loading image with code
from IPython.display import Image
Image(filename='C:/Users/mateus_silva1/Python_Jupyter_Studies/images/append and extend.PNG')
```




![png](output_43_0.png)




```python
#lists have order, are iterable, are indexable, and can be changed
x = [1,2,3,"a","b","b",1]
print(type(x))
print(x)
#append
x.append("c")
print("printing after appending one element", x)
y = [5,6,7]
x = x+y
print("printing after summing ",x)
#if append x to y, result will be different then summing (concatenating)
x.append(y)
print("printing after appending two lists", x)
#concatenate with EXTEND method: you can use + signal or concat method
x1 = [1,2,3]
x2 = [3,2,1]
#can't pass to a third list
x1.extend(x2)
print ("printing after appending with EXTEND method: ",x1)
```

    <class 'list'>
    [1, 2, 3, 'a', 'b', 'b', 1]
    printing after appending one element [1, 2, 3, 'a', 'b', 'b', 1, 'c']
    printing after summing  [1, 2, 3, 'a', 'b', 'b', 1, 'c', 5, 6, 7]
    printing after appending two lists [1, 2, 3, 'a', 'b', 'b', 1, 'c', 5, 6, 7, [5, 6, 7]]
    printing after appending with EXTEND method:  [1, 2, 3, 3, 2, 1]
    

## List Comprehension:


```python
#ex1
#WITHOUT LIST COMPREHENSION (FROM)
my_list = []
for number in range(0,1000):
    if number %2 == 0:
        my_list.append(number)
#ex1
#TO: WITH LIST COMPREHENSION
my_list = [number for number in range(0,1000) if number % 2 == 0]


#ex2 with list comprehension #correct_answer is an array with all possible combinations ()
#first 2 letters should be LETTER and LAST TWO numbers
lowercase = 'abcdefghijklmnopqrstuvwxyz'
digits = '0123456789'

correct_answer = [a+b+c+d for a in lowercase for b in lowercase for c in digits for d in digits]
print("showing 50 first combinations \n{}".format(correct_answer[:50]))



#ex2 with for
test = []
for a in lowercase:
    for b in lowercase:
        for c in digits:
            for d in digits:
                test.append(a+b+c+d)
number_of_combinations = len(test)
print ("number of combinations is {} \n".format(number_of_combinations))
```

    showing 50 first combinations 
    ['aa00', 'aa01', 'aa02', 'aa03', 'aa04', 'aa05', 'aa06', 'aa07', 'aa08', 'aa09', 'aa10', 'aa11', 'aa12', 'aa13', 'aa14', 'aa15', 'aa16', 'aa17', 'aa18', 'aa19', 'aa20', 'aa21', 'aa22', 'aa23', 'aa24', 'aa25', 'aa26', 'aa27', 'aa28', 'aa29', 'aa30', 'aa31', 'aa32', 'aa33', 'aa34', 'aa35', 'aa36', 'aa37', 'aa38', 'aa39', 'aa40', 'aa41', 'aa42', 'aa43', 'aa44', 'aa45', 'aa46', 'aa47', 'aa48', 'aa49']
    number of combinations is 67600 
    
    


```python
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text

data['body_text_nostop'] = data['body_text_tokenized'].apply(lambda x: remove_stopwords(x))
```

## Tuples


```python
#tuples can't be changed (immutable), and don't have order
my_tuple = (1,2,3,3)
type(my_tuple)
print(my_tuple)
```

    (1, 2, 3, 3)
    

## Strings


```python
#loading image with code
from IPython.display import Image
Image(filename='C:/Users/mateus_silva1/Python_Jupyter_Studies/images/string_methods.PNG')
```




![png](output_51_0.png)




```python
string = "my name is "
name = "mateus"
age = 20
print (string + name + "and I am", age)
#other way to put strings and numbers, with format method
string_to_print = string + name + "and I am {}"
print (string_to_print.format(age))
```

    my name is mateusand I am 20
    my name is mateusand I am 20
    

### more on formatting strings


```python
name = "John"
price = 10.23241
txt = "His name is {} and he spent {:.2f} dollars with food"
print(txt.format(name, price))
#another way of doing it
txt = "His name is {0} and {0} he spent {1:.2f} dollars with food"
print (txt.format(name,price))
```

    His name is John and he spent 10.23 dollars with food
    His name is John and John he spent 10.23 dollars with food
    

### check if text is in string, and where it is?


```python
#find piece of string in string with IN
text = "I have nothing to do at this moment but studying python"
x = "at this moment" in text
print(x)
y = text.find("at this moment")
print (y)
```

    True
    21
    

### Strip (take white spaces out) and Split (separate by a value) strings 


```python
text = "  this is the string that I want to separate by spaces and also strip it from white spaces  "
strip_text = text.strip()
print (strip_text)
split_text = strip_text.split(" ")
for item in split_text:
    print (item)
print("\nthe number of words in the string is: ", len(split_text)) 
# of letters in the first word
number_letters = len(split_text[1])
print ("the number of letters in the 2nd word is: ", number_letters)
#now iterating
count = 0
for count in range(len(split_text[0])):
    count += 1
print ("count of letters with for is: ", count)
```

    this is the string that I want to separate by spaces and also strip it from white spaces
    this
    is
    the
    string
    that
    I
    want
    to
    separate
    by
    spaces
    and
    also
    strip
    it
    from
    white
    spaces
    
    the number of words in the string is:  18
    the number of letters in the 2nd word is:  2
    count of letters with for is:  4
    

## Sets: non-ordered, not repeatable elements (good for categories)


```python
array_to_set = [1,2,3,3, "car", "car", "bike"]
set_from_array = set(array_to_set)
print(set_from_array)
```

    {1, 2, 3, 'bike', 'car'}
    

# Dictionaries


```python
first_dic = {
            "names": ["john", "eden", "health"],
            "prices": [10, 20, 30],
            "time_used": [5,6,8]
            }
print(first_dic)
```

    {'names': ['john', 'eden', 'health'], 'prices': [10, 20, 30], 'time_used': [5, 6, 8]}
    

#### Exercise with Dictionaries: Iterate through them and do averages with for loop


```python
import csv
%precision 2


url = 'C:/Users/mateus_silva1/Documents/Data Science/Python Scripts/mtcars.csv'
with open(url) as csvfile:
    cars = list(csv.DictReader(csvfile)) #OBJECT IS LIST OF DICTIONARIES
len(cars)
#GET SUM OF ONE OF THE COLUMNS OF DICT
avg_mpg = sum(float(dict_i["mpg"]) for dict_i in mpg) / len(cars)
print ('avg of mpg is {:.2f}'.format(avg_mpg))


#AVERAGE of Categorical field:
#use set and for inside for
set_cyl = set((dict_items["cyl"]) for dict_items in cars)
append_results = []
#also: iterating thru dictionaries with simple types
for cylz in set_cyl:
    for dict_items in cars:
        if (dict_items["cyl"] == cylz):
            sum_cyl += float(dict_items["cyl"])
            count_cyl += 1
    append_results.append((cylz, sum_cyl/count_cyl))    
append_results
```

    avg of mpg is 20.09
    




    [('6', 6.17), ('4', 5.96), ('8', 6.19)]



# Classes and Objects

## create class and function to define it's name


```python
####CLASS AND OBJECTS
class Books:
    name=[]
    price=float
    def name_func(self, written_name):
        name = written_name
        return name
```

### create objects


```python
#setting name through defined function
Lord_rings = Books()
Harry = Books()
Lord_rings.name = Lord_rings.name_func("Lord of the rings")
print (Lord_rings.name)
Harry.name = Harry.name_func("Harry Potter")
print (Harry.name)
type(Harry)
#setting name here without function
Laranja = Books()
Laranja.name = "Laranja mecanica"
Laranja.price = 100.25
print ("the name of this book is: ", Laranja.name, " and the price is: ", Laranja.price)
#looping through objects inside a class
print(type(Books)) #class type
```

    Lord of the rings
    Harry Potter
    the name of this book is:  Laranja mecanica  and the price is:  100.25
    <class 'type'>
    

### old and new method to initialize variables with a construct


```python
#old method in python2
class Point:
    #def __init__ is a construct to initializa variables in a class
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z
#using dataclass (available in python 3.7)
#easier to initialize
from dataclasses import dataclass
@dataclass
class Books1:
    name=[]
    price=float
    def name_func(self, written_name):
        name = written_name
        return name

```


```python
#setting name through defined function with @dataclass
Laranja = Books1()
Laranja.name = "Laranja mecanica"
Laranja.price = 100.25
print ("the name of this book is: ", Laranja.name, " and the price is: ", Laranja.price)
#looping through objects inside a class
print(type(Books)) #class type
```

    Lord of the rings
    Harry Potter
    the name of this book is:  Laranja mecanica  and the price is:  100.25
    <class 'type'>
    

# NUMPY ARRAYS

### NUMPY CHEATSHEET URL: https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf

### creating numpy arrays: var = np.array([ ]) 

### var = np.arange(fromx, toy, spacedby): ex var = np.arange(0,30,2) 

### np.reshape(nofrows,nofcolumns) ... np.linspace(fromx,toy,inznumbers)... np.ones... np.risize (similar to reshape but change the array), np.reshape (reshape without changing the variable), np.ones((x,y)), np.zeros, np.eye(3) - identity matrix) - AND COPY(if you change a subset of array the original array will change, so, copy avoids that)


```python
import numpy as np
import pandas as pd
```


```python
twodimarray = np.array(
    [
        [1,3,5,7],
        [2,4,6,8]
    ]
)
#for with matrix, see result to understand it
for array in twodimarray:
    print("i array print: ", array)
    for item in array:
        print ("item in array print: ",item)
        
threedimarray = np.array(
    [
        [1,3,5,7],
        [2,4,6,8],
        [9,10,11,12]
    ]
)
print ("\n\nprinting array", threedimarray)
print("shape of array is: ",threedimarray.shape)#3 lines and 4 columns
print("number of dimensions: ", threedimarray.ndim)
print("# of elements (size): ",threedimarray.size)
print("type is: ", type(threedimarray))        
    
```

    i array print:  [1 3 5 7]
    item in array print:  1
    item in array print:  3
    item in array print:  5
    item in array print:  7
    i array print:  [2 4 6 8]
    item in array print:  2
    item in array print:  4
    item in array print:  6
    item in array print:  8
    
    
    printing array [[ 1  3  5  7]
     [ 2  4  6  8]
     [ 9 10 11 12]]
    shape of array is:  (3, 4)
    number of dimensions:  2
    # of elements (size):  12
    type is:  <class 'numpy.ndarray'>
    

### Creating NP Arrays: (randon, reshape, arange, etc)


```python
array_arrange = np.arange(0,5,0.5) #create from x to y with spacing of z (x,y,z)
print("printing arrange:\n{}\n".format(array_arrange))
np_linspace_array = np.linspace(1,10,20) #create from x to y with Z # of elements
print("printing np_linspace_array is \n{}".format(np_linspace_array))
random_array = np.random.random((5,5))
print("printing randon 5x5 array \n{}".format(random_array))
```

    printing arrange:
    [0.  0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5]
    
    printing np_linspace_array is 
    [ 1.          1.47368421  1.94736842  2.42105263  2.89473684  3.36842105
      3.84210526  4.31578947  4.78947368  5.26315789  5.73684211  6.21052632
      6.68421053  7.15789474  7.63157895  8.10526316  8.57894737  9.05263158
      9.52631579 10.        ]
    printing randon 5x5 array 
    [[0.67968627 0.26859754 0.53644486 0.08759563 0.44935964]
     [0.84394771 0.01337892 0.31026094 0.51119629 0.51924   ]
     [0.72161946 0.56252718 0.53359998 0.84337145 0.1250302 ]
     [0.59193794 0.95468606 0.71244646 0.74640217 0.17983672]
     [0.29584518 0.35390154 0.78618213 0.45127921 0.75308431]]
    

#### np.arange


```python
array = np.arange(0,10,2)
array
```




    array([0, 2, 4, 6, 8])




```python
#from list to array
list =[1,2,3]
array_np = np.array(list)
```

#### Copy array to another array + randon and reshape


```python
#not correct, changing root array
z = np.random.randint(0,10,(4,3))
z1 = z
z1[:,1] = 1
print(z) #it change the 2nd columns values to 1 even for z, and I just changed z1

#correct way, not changing root array
r = np.arange(0,10,1)
r_copied = r.copy()
print("\nr_copied is \n{}".format(r_copied))
```

    [[1 1 2]
     [9 1 5]
     [2 1 4]
     [7 1 8]]
    
    r_copied is 
    [0 1 2 3 4 5 6 7 8 9]
    

#### Same Happens to DATAFRAMES, when you change a new dataframe you change the old one, should use COPY then


```python
z = np.arange(0,10,1).reshape(2,5)
df = pd.DataFrame(z, columns=["a","b","c","d","e"], index=[0,1])
df_1 = df
df_1["b"] = 1
df_1
print("printing df \n {} \n printing df_1 \n {}".format(df,df_1))
```

    printing df 
        a  b  c  d  e
    0  0  1  2  3  4
    1  5  1  7  8  9 
     printing df_1 
        a  b  c  d  e
    0  0  1  2  3  4
    1  5  1  7  8  9
    


```python

```


```python

```


```python
#create array from 0 to x, create random array, and create random matrix

array_1 = np.arange(10) #creating np array from 0 to 9
np.random.seed(25) #setting 25 the number until the randon change
randomnpmtx1 = np.random.random((3,3)) #creating randon matrix np with random values
print ("first nparray is: ", array_1, "\nrandon array is: ", np.round(randomnpmtx1,2)) #\n do break line
randomarray = np.random.random(3)
print ("randon array is: ", randomarray)
```

    first nparray is:  [0 1 2 3 4 5 6 7 8 9] 
    randon array is:  [[0.87 0.58 0.28]
     [0.19 0.41 0.12]
     [0.68 0.44 0.56]]
    randon array is:  [0.36708032 0.40236573 0.1130407 ]
    


```python

```

# PANDAS DATAFRAMES

## Pandas CheatSheet: 
1) http://datacamp-community-prod.s3.amazonaws.com/dbed353d-2757-4617-8206-8767ab379ab3<br></br>
2.a) https://miro.medium.com/max/3820/1*YhTbz8b8Svi22wNVvqzneg.jpeg <br></br>
2.b) https://miro.medium.com/max/3820/1*3-mTHM3ejorJwLnjuIpVNQ.jpeg<br></br>



```python
import pandas as pd
import numpy as np
```

#### Creating Series (1 dim pandas object)


```python
a = [1,2,3,"a"]
series_obj = pd.Series(a, index=[1,2,3,5])
series_obj
```




    1    1
    2    2
    3    3
    5    a
    dtype: object



# Creating DataFrames with Pandas

##### creating dataframe from a csv file, put columns (cars.columns = ['col1','col2','etc'] and index(easy way)  in two ways : index_col = 0 on load or cars.set_index (more complicated)


```python
#LOADING CARs DATA with index_col = [column]
address = 'C:/Users/mateus_silva1/Documents/Data Science/Python Scripts/mtcars.csv'
cars = pd.read_csv(address, index_col = 0)
cars.columns = ['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
cars.head()


#LOADING CARs DATA with cars.set_index
address = 'C:/Users/mateus_silva1/Documents/Data Science/Python Scripts/mtcars.csv'
cars = pd.read_csv(address)
cars.columns = ['car_index', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
cars.set_index(['car_index'], drop=True, append=False, inplace=True, verify_integrity=False) #making car
cars.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cyl</th>
      <th>disp</th>
      <th>hp</th>
      <th>drat</th>
      <th>wt</th>
      <th>qsec</th>
      <th>vs</th>
      <th>am</th>
      <th>gear</th>
      <th>carb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Mazda RX4</th>
      <td>21.0</td>
      <td>6</td>
      <td>160.0</td>
      <td>110</td>
      <td>3.90</td>
      <td>2.620</td>
      <td>16.46</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Mazda RX4 Wag</th>
      <td>21.0</td>
      <td>6</td>
      <td>160.0</td>
      <td>110</td>
      <td>3.90</td>
      <td>2.875</td>
      <td>17.02</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Datsun 710</th>
      <td>22.8</td>
      <td>4</td>
      <td>108.0</td>
      <td>93</td>
      <td>3.85</td>
      <td>2.320</td>
      <td>18.61</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Hornet 4 Drive</th>
      <td>21.4</td>
      <td>6</td>
      <td>258.0</td>
      <td>110</td>
      <td>3.08</td>
      <td>3.215</td>
      <td>19.44</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Hornet Sportabout</th>
      <td>18.7</td>
      <td>8</td>
      <td>360.0</td>
      <td>175</td>
      <td>3.15</td>
      <td>3.440</td>
      <td>17.02</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



import os
import pandas as pd
current_dir = os.getcwd()
print ("current directory is: ", current_dir)
dir_to_list = "C:/Users/mateus_silva1/Documents/Data Science/Python Scripts"
os.listdir(dir_to_list)
os.chdir("C:/Users/mateus_silva1/Documents/Data Science/Python Scripts") #changing work directory
df = pd.read_csv('mtcars.csv', index_col=0)
df.iloc[0:5,0:3]
#READ EXCEL FILE df = pd.read_excel('File.xlsx', sheetname='Sheet1')

## Creating DataFrames with Different Modes:


```python
#WITH LISTS OF LISTS
# initialize list of lists 
data = [
    ['tom', 10], 
    ['nick', 15], 
    ['juli', 14]
       ]       
# Create the pandas DataFrame 
df = pd.DataFrame(data, columns = ['Name', 'Age'], index = [2,1,0]) 




#WITH LIST OF LISTS
# initialize list of lists 
data = [
    ['tom', 10], 
    ['nick', 15], 
    ['juli', 14]
       ]   
# Create the pandas DataFrame 
df = pd.DataFrame(data, columns = ['Name', 'Age'], index = [2,1,0]) 




#WITH DATES AS INDEX AND NP ARRAYS
my_dates_index = pd.date_range('20190101', periods=8)
my_dates_index
sample_numpy_data = np.array(np.arange(40)).reshape(8,5)
sample_numpy_data
sample_df = pd.DataFrame(sample_numpy_data, index=my_dates_index, columns=["A","B","C","D","E"])




#WITH DICTIONARY
df_from_dictionary = pd.DataFrame({
        'float':1.,
        'time':pd.Timestamp('20160825'),
        'series':pd.Series(1,index=list(range(4)),dtype='float32'),
        'array':np.array([3]*4,dtype='int32'),
        'categories': pd.Categorical(['test','train','taxes','tools']),
        'dull' : 'boring data'
        })



#FROM DICTIONARY WITH INDEX
# Python code demonstrate creating  
# DataFrame from dict narray / lists  
# By default addresses.    
import pandas as pd 
# intialise data with dict. and lists. 
#it takes the keys from dict to be columns
data = {
        'Name':['Tom', 'nick', 'krish', 'jack'], 
        'Age':[20, 21, 19, 18]
       }
# Create DataFrame 
df = pd.DataFrame(data, index = ('top 1','top 2','top 3', 'top 4'))




#WITH LIST OF DICTIONARIES  
# Intitialise lists data. 
data = [
    {'a': 1, 'b': 2}, 
    {'a': 5, 'b': 10, 'c': 20},
    {'f': 1, 'b': 2}
        ] 
# With two column indices, values same  
# as dictionary keys 
df1 = pd.DataFrame(data, index =['first', 'second','third'], columns =['a', 'b','c','b1'])
#if I add the F column at the dataframe, the value 1 would appear in the third row
#b1 doesn't exist at the dict, that's why the null





#FROM DICT OF PD.SERIES
# Python code demonstrate creating 
# Pandas Dataframe from Dicts of series.
# Intialise data to Dicts of series. 
d = { 'one' : pd.Series([10, 20, 30, 40], index =['a', 'b', 'c', 'd']), 
      'two' : pd.Series([10, 20, 30, 40], index =['a', 'b', 'c', 'd'])}  
# creates Dataframe. 
df = pd.DataFrame(d) 





#WITH LISTS USING ZIP
# Python program to demonstrate creating  
# pandas Datadaframe from lists using zip.      
import pandas as pd  
# List1  
Name = ['tom', 'krish', 'nick', 'juli']  
# List2  
Age = [25, 30, 26, 22]  
# get the list of tuples from two lists.  
# and merge them by using zip().  
list_of_tuples = list(zip(Name, Age))     
# Assign data to tuples.  
list_of_tuples   
# Converting lists of tuples into  
# pandas Dataframe.  
df = pd.DataFrame(list_of_tuples, columns = ['Name', 'Age'])  
```

### PANDAS METHODS 


```python
#loading image with code
from IPython.display import Image
Image(filename='C:/Users/mateus_silva1/Python_Jupyter_Studies/images/pandas_methods.PNG')
#LINK TO PAGE: https://www.geeksforgeeks.org/python-pandas-dataframe/#Basics4 
```




![png](output_105_0.png)



#### RESET AND CHANGE NAMES OF INDEX n COLUMNS: reset_index, index.name, set_index; COLUMNS (df.columns)

#### Can use multi index (hierarchical, state and city, ex) to set_index, but then to use .loc has to use 2 arguments for the index names


```python
import pandas as pd
import numpy as np
#creating dict
dict_to_df = {
                'name': ["tom", "john", "patrick"],
                'country': ["usa","australia","canada"],
                'city': ["austin","sidney","montreal"]                
             }
#creating df
df = pd.DataFrame(dict_to_df, index = np.arange(1,4))
#reseting index and changing index name
df.reset_index(drop=True, inplace=True) #WITH INPLACE EQUALS TO TRUE THERE'S NO NEED OF CREATING OTHER DF
df.index.name = "index_column" 
df.columns = ["aa","bb","cc"] #CHANGING COLUMN NAMES
print("printing df \n\n {}".format(df))
print("df columns are \n\n{}".format(df.columns))

#creating df and reseting index and changing index name
df1 = pd.DataFrame(dict_to_df, index = np.arange(1,4))
df2 = df1.reset_index(drop=True) #no need for inplace since I'm putting on other DF
df2.index.name="changed index" #changing index name
print("printing df2 \n\n {}".format(df2))

######set index
df3 = pd.DataFrame(dict_to_df, index = np.arange(1,4))
df3.set_index("name", drop=True, inplace=True) #repeating index because drop is false.
print("printing df3 \n\n {}".format(df3))
```

    printing df 
    
                        aa         bb        cc
    index_column                              
    0                 tom        usa    austin
    1                john  australia    sidney
    2             patrick     canada  montreal
    df columns are 
    
    Index(['aa', 'bb', 'cc'], dtype='object')
    printing df2 
    
                       name    country      city
    changed index                              
    0                  tom        usa    austin
    1                 john  australia    sidney
    2              patrick     canada  montreal
    printing df3 
    
                country      city
    name                        
    tom            usa    austin
    john     australia    sidney
    patrick     canada  montreal
    

### DF.RENAME, and with AXIS - FOR INDEX AND COLUMNS


```python
#COLUMNS FIRST
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
df.rename(columns={"A": "a", "B": "c"})
#INDEX NOW
df.rename(index={0: "x", 1: "y", 2: "z"})

#AXIS 
df.rename(str.lower, axis='columns')
df.rename({1: 2, 2: 4}, axis='index')

```

### Pandas Boolean Mask and WHERE Function


```python
import pandas as pd
#LOADING 0lympics data with index_col = [column]
address = 'C:/Users/mateus_silva1/Documents/Data Science/Python Scripts/olympics.csv'
df = pd.read_csv(address, index_col = 0, skiprows=1)

#cleanning (changing) column names
for col in df.columns:
    if col[:2] == '01':
        df.rename(columns={col:'Gold'+col[4:]}, inplace = True)
    if col[:2] == '02':
        df.rename(columns={col:'Silver'+col[4:]}, inplace = True)
    if col[:2] == '03':
        df.rename(columns={col:'Bronze'+col[4:]}, inplace = True)  
    if col[:1] == '№':
        df.rename(columns={col:'#'+col[2:]}, inplace = True)
        
#BOOLEAN MASKS
only_gold = df.where(df['Gold'] > 0) #with where explicitly
print("df only_gold head is: \n {}".format(only_gold.head()))
print("\n\n number of gold medals is {}".format(only_gold['Gold'].count()))
#without where: more concise
only_gold = df[df['Gold'] > 0]
only_gold.head()

######
##### & AND | are AND and OR operators
# DF WITH COUNTRIES WHO GOT GOLDEN MEDAL INS SUMMER AND WINTER OLYMPICS
df_gold_summer_winter = df[(df['Gold']>0) | (df['Gold.1']>0)]
#print # of countries
print("number of countries with golden in summer and winter olympic games is (using len) {}"
      .format(len(df[(df['Gold']>0) & (df['Gold.1']>0)])))
```

    df only_gold head is: 
                              #Summer  Gold  Silver  Bronze  Total  #Winter  \
    Afghanistan (AFG)            NaN   NaN     NaN     NaN    NaN      NaN   
    Algeria (ALG)               12.0   5.0     2.0     8.0   15.0      3.0   
    Argentina (ARG)             23.0  18.0    24.0    28.0   70.0     18.0   
    Armenia (ARM)                5.0   1.0     2.0     9.0   12.0      6.0   
    Australasia (ANZ) [ANZ]      2.0   3.0     4.0     5.0   12.0      0.0   
    
                             Gold.1  Silver.1  Bronze.1  Total.1  #Games  Gold.2  \
    Afghanistan (AFG)           NaN       NaN       NaN      NaN     NaN     NaN   
    Algeria (ALG)               0.0       0.0       0.0      0.0    15.0     5.0   
    Argentina (ARG)             0.0       0.0       0.0      0.0    41.0    18.0   
    Armenia (ARM)               0.0       0.0       0.0      0.0    11.0     1.0   
    Australasia (ANZ) [ANZ]     0.0       0.0       0.0      0.0     2.0     3.0   
    
                             Silver.2  Bronze.2  Combined total  
    Afghanistan (AFG)             NaN       NaN             NaN  
    Algeria (ALG)                 2.0       8.0            15.0  
    Argentina (ARG)              24.0      28.0            70.0  
    Armenia (ARM)                 2.0       9.0            12.0  
    Australasia (ANZ) [ANZ]       4.0       5.0            12.0  
    
    
     number of gold medals is 100
    number of countries with golden in summer and winter olympic games is (using len) 37
    

#### Filtering DataFrames with PAndas


```python
#creating dict
dict_to_df = {
                'name':    ["tom", "john", "patrick"],
                'country': ["usa","australia","canada"],
                'city':    ["austin","sidney","montreal"],
                'price':   [500,100,3]
             }
#creating df
df = pd.DataFrame(dict_to_df, index = np.arange(1,4))
df_bigger_eq_100 = df[df>=100] #FILTERING
df_tom = df[df=="tom"] #FILTERING
print("Print df_bigger: \n\n {} \n\n and df_tom_filter: \n\n {}".format(df_bigger_eq_100,df_tom))



#OTHER
#EXAMPLE OF FILTERING
url = 
fullCorpus = pd.read_csv(url, sep="\t", header=None, names=['label','body_text']) #names are to set columns
# another way of putting column names
#fullCorpus.columns = ['label', 'body_text']
fullCorpus.head()
#EXPLORE THE DATASET
#what is the shape of the dataset?
fullCorpus.shape
#or
print ("Input data has {} rows and {} columns".format(len(fullCorpus),len(fullCorpus.columns)))
#how many spam/ham are there?
print ("out of {} rows, {} rows are spam, and {} rows are ham".format(len(fullCorpus),
        len(fullCorpus[fullCorpus.label=='spam']), #2 WAYS OF FILTERING WITH PANDAS
        len(fullCorpus[fullCorpus['label']=='ham']))) #2 WAYS OF FILTERING WITH PANDAS
```


      File "<ipython-input-6-79185543da67>", line 18
        url =
              ^
    SyntaxError: invalid syntax
    


### GroupBY with Pandas


```python
grouped_df = pd.DataFrame(df.groupby('name')['amount'].sum().sort_values(ascending = False))
```

### Concatenate with Pandas


```python
#loading image with code
from IPython.display import Image
Image(filename='C:/Users/mateus_silva1/Python_Jupyter_Studies/images/concatenate.PNG')
```




![png](output_118_0.png)




```python

```


```python

```

### Merge with Pandas; join is similar to Merge but focused on the index only, not on any other column from the df


```python
#loading image with code
from IPython.display import Image
Image(filename='C:/Users/mateus_silva1/Python_Jupyter_Studies/images/merge.PNG')
```




![png](output_122_0.png)




```python
#creating from list of lists
data = [
    ['tom', 10, 'wild'], 
    ['nick', 15, 'loving'], 
    ['juli', 14, 'sunny'],
    ['juli', 14, 'sunny'],
    ['juli', 14, 'sunny'],
    ['tom', 20, 'wild'],
       ]
#creating DF
df = pd.DataFrame(data,columns=['name','amount','type'])
print("df without gruping: \n",df)
#grouping DF and putting in a new DF
grouped_df = pd.DataFrame(df.groupby(['name','type'])['amount'].sum().sort_values(ascending = False))
print("df with grouping: \n",grouped_df)

#start merging
#had to create var without the values to merge, otherwise it wouldn't be grouped
to_merge= [
    ['tom', 'furious'],
    ['juli','awesome'],
    ['nick','Bad']
          ]
to_merge_df = pd.DataFrame(to_merge, columns=['name','humor'])
###############################################################
merged_df = pd.merge(grouped_df, to_merge_df, on='name')
print("merged DF: \n", merged_df)
```

    df without gruping: 
        name  amount    type
    0   tom      10    wild
    1  nick      15  loving
    2  juli      14   sunny
    3  juli      14   sunny
    4  juli      14   sunny
    5   tom      20    wild
    df with grouping: 
                  amount
    name type          
    juli sunny       42
    tom  wild        30
    nick loving      15
    merged DF: 
        name  amount    humor
    0  juli      42  awesome
    1   tom      30  furious
    2  nick      15      Bad
    

#### Pivot Tables with Pandas


```python

```

#### Creating a DF to play with some methods and slicers 


```python
import pandas as pd
import numpy as np
#DF_obj = DataFrame(np.arange(36).reshape(6,6),index = ['row1','row2','row3','row4','row5','row6'],
#columns=['column1','column2','column3','column4','column5','column6'])
#############CREATING DATA FRAME WITH RANDOM FROM NUMPY
df = pd.DataFrame(np.random.rand(36).round(decimals=2).reshape((6,6)), 
                      index = ['row1','row2','row3','row4','row5','row6'],
                      columns=['column1','column2','column3','column4','column5','column6'])
df.index
df.columns
print(df.iloc[:,1:3]) #ILOC INDEXES BASED COLUMNS POSITION NUMBERS
print(df.loc[:,['column1', 'column3']]) #BASED IN COLUMN names
#df.ix works with both position numbers and column names
#reseting index and also dropping the column that was index
df1 = df.reset_index().drop(["index"], axis = 1, inplace= False) #inplace is false is to create another df
#inplace is true change the df itself, so you wouldn't create the DF1, you would do the method like df.reset_index...
print ("df after reseting index: \n", df1)
```

          column2  column3
    row1     0.16     0.55
    row2     0.83     0.90
    row3     0.72     0.93
    row4     0.52     0.17
    row5     0.13     0.01
    row6     0.11     0.96
          column1  column3
    row1     0.85     0.55
    row2     0.41     0.90
    row3     0.05     0.93
    row4     0.26     0.17
    row5     0.07     0.01
    row6     0.88     0.96
    df after reseting index: 
        column1  column2  column3  column4  column5  column6
    0     0.85     0.16     0.55     0.22     0.26     0.36
    1     0.41     0.83     0.90     0.52     0.50     0.22
    2     0.05     0.72     0.93     0.06     0.07     0.36
    3     0.26     0.52     0.17     0.80     0.95     0.22
    4     0.07     0.13     0.01     0.28     0.51     0.71
    5     0.88     0.11     0.96     0.79     0.03     0.46
    

### iterating over Pandas DataFrame - couldn't do it, jaja


```python
#JUST CREATING A DATAFRAME LIKE ABOVE CELL
df_var = pd.DataFrame(np.random.rand(36).round(decimals=2).reshape((6,6)), 
                      index = ['row1','row2','row3','row4','row5','row6'],
                      columns=['column1','column2','column3','column4','column5','column6'])
print ("this is the df: \n", df_var)
#####################################################
#iterating over rows using iterrows() function  
for row in df_var.iterrows(): 
    print("iteration row: \n",row) 
    for colums in row:
        print("iteration column: \n", colums)
```

    this is the df: 
           column1  column2  column3  column4  column5  column6
    row1     0.86     0.95     0.19     0.16     0.95     0.99
    row2     0.65     0.73     0.18     0.47     0.68     0.69
    row3     0.34     0.03     0.94     0.46     0.71     0.78
    row4     0.17     0.87     0.17     0.29     0.79     0.59
    row5     0.74     0.91     0.24     0.84     0.09     0.11
    row6     0.49     0.38     0.52     0.01     0.12     0.41
    iteration row: 
     ('row1', column1    0.86
    column2    0.95
    column3    0.19
    column4    0.16
    column5    0.95
    column6    0.99
    Name: row1, dtype: float64)
    iteration column: 
     row1
    iteration column: 
     column1    0.86
    column2    0.95
    column3    0.19
    column4    0.16
    column5    0.95
    column6    0.99
    Name: row1, dtype: float64
    iteration row: 
     ('row2', column1    0.65
    column2    0.73
    column3    0.18
    column4    0.47
    column5    0.68
    column6    0.69
    Name: row2, dtype: float64)
    iteration column: 
     row2
    iteration column: 
     column1    0.65
    column2    0.73
    column3    0.18
    column4    0.47
    column5    0.68
    column6    0.69
    Name: row2, dtype: float64
    iteration row: 
     ('row3', column1    0.34
    column2    0.03
    column3    0.94
    column4    0.46
    column5    0.71
    column6    0.78
    Name: row3, dtype: float64)
    iteration column: 
     row3
    iteration column: 
     column1    0.34
    column2    0.03
    column3    0.94
    column4    0.46
    column5    0.71
    column6    0.78
    Name: row3, dtype: float64
    iteration row: 
     ('row4', column1    0.17
    column2    0.87
    column3    0.17
    column4    0.29
    column5    0.79
    column6    0.59
    Name: row4, dtype: float64)
    iteration column: 
     row4
    iteration column: 
     column1    0.17
    column2    0.87
    column3    0.17
    column4    0.29
    column5    0.79
    column6    0.59
    Name: row4, dtype: float64
    iteration row: 
     ('row5', column1    0.74
    column2    0.91
    column3    0.24
    column4    0.84
    column5    0.09
    column6    0.11
    Name: row5, dtype: float64)
    iteration column: 
     row5
    iteration column: 
     column1    0.74
    column2    0.91
    column3    0.24
    column4    0.84
    column5    0.09
    column6    0.11
    Name: row5, dtype: float64
    iteration row: 
     ('row6', column1    0.49
    column2    0.38
    column3    0.52
    column4    0.01
    column5    0.12
    column6    0.41
    Name: row6, dtype: float64)
    iteration column: 
     row6
    iteration column: 
     column1    0.49
    column2    0.38
    column3    0.52
    column4    0.01
    column5    0.12
    column6    0.41
    Name: row6, dtype: float64
    


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

## Data Mining, Tokenization, Stop Words, breaking strings, etc with Lambda Functions


```python
import pandas as pd
pd.set_option('display.max_colwidth', 100)

data = pd.read_csv("C:/Users/mateus_silva1/Documents/Data Science/Python Scripts/SMSSpamCollection.tsv", sep='\t', header=None)
data.columns = ['label', 'body_text']

data.head()
import string
string.punctuation
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>body_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>I've been searching for the right words to thank you for this breather. I promise i wont take yo...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives around here though</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>Even my brother is not like to speak with me. They treat me like aids patent.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>I HAVE A DATE ON SUNDAY WITH WILL!!</td>
    </tr>
  </tbody>
</table>
</div>




```python
#go through each character, if it's punctuation, take it out 
def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct

data['body_text_clean'] = data['body_text'].apply(lambda x: remove_punct(x))
data.head()



import re

def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens

data['body_text_tokenized'] = data['body_text_clean'].apply(lambda x: tokenize(x.lower()))

data.head()


def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text

data['body_text_nostop'] = data['body_text_tokenized'].apply(lambda x: remove_stopwords(x))
data

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
