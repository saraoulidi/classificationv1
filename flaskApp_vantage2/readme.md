This folder allows you to run your prediction file ( p.py ) with a rest API using Flask and swagger.

To run this use: 

```bash
FLASK_APP=app.py flask run
```

ou bien


```bash
python3.6 -m flask run
```


> Make sure that :

The file pred.py contains your model name generated from ```t.py```. In this example, we used ```sansAug2```.

your Dataset folder & your model are in the same dir as ```pred.py``` & ```app.py``` files

