default:all
	@echo 1 >/dev/null
all:
	@cp -r traffic_light_detection/* run/
	@make -c run
run:
	python3 run/server.py