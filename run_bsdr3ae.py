from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "bsdr3ae"
    tasks = {
        "algorithms" : ["bsdr3ae"],
        "datasets": ["indian_pines"],
        "target_sizes" : list(range(2,31))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=False, test=False)
    summary, details = ev.evaluate()
