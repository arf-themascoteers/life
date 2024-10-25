from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "2"
    tasks = {
        "algorithms" : ["pcal","linspacer","random","c1","bsnet","bnc","mcuve"],
        "datasets": ["indian_pines"],
        "target_sizes" : [5,10,15,20,25,30]
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=False, verbose=False)
    summary, details = ev.evaluate()
