from task_runner import TaskRunner
import plotter_classification
import clean_up

if __name__ == '__main__':
    clean_up.do_it()
    tag = "p23"
    tasks = {
        "algorithms" : ["c3"],
        "datasets": ["ghisaconus_health"],
        "target_sizes" : list(range(2,31))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=True, test=False)
    summary, details = ev.evaluate()
    plotter_classification.plot_combined(sources=["p21","p23"],only_algorithms=["bsnet","bnc","c1","c3"], only_datasets=["ghisaconus_health"])