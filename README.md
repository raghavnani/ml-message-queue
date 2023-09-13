Get started 
------------

1. Rename `.env.dist` to `env`.
2. `$ docker-compose up --scale worker=3`

Classify text
-------------
POST to `/api/task/send-task/message_queue.classify` of flower dashboard with args-array of respective method in body. 

`curl --location --request POST 'http://localhost:5555/api/task/send-task/message_queue.classify'
--data-raw '{
	"args": ["test 123", "default"]
}'`


Access Celery dashboard
----------------
http://localhost:5555/
