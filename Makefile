.PHONY: gm gitmerge


gm gitmerge:
	@git switch master && git merge --no-ff -m "merge dev" dev && git push && git switch dev


