from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "j1"
    tasks = {
        "algorithms" : ["mcuve", "spa", "cars", "rf"],
        "datasets": ["ghisaconus"],
        "target_sizes" : list(range(5,6))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=False, verbose=False, test=False)
    summary, details = ev.evaluate()
