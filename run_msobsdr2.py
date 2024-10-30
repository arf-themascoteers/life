from task_runner import TaskRunner

#All algorithms for lucas except for multi-bsdr
if __name__ == '__main__':
    tag = "mb"
    tasks = {
        "algorithms" : ["msobsdr2"],
        "datasets": ["indian_pines"],
        "target_sizes" : list(range(2,31))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=False, test=False)
    summary, details = ev.evaluate()
