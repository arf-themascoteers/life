from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "msobsdr3ae"
    tasks = {
        "algorithms" : ["msobsdr3ae"],
        "datasets": ["indian_pines"],
        "target_sizes" : list(range(10,11))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=False, test=False)
    summary, details = ev.evaluate()
