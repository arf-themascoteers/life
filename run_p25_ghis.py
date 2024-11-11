from task_runner import TaskRunner
import plotter_classification
import clean_up

if __name__ == '__main__':
    clean_up.do_it()
    tag = "p25"
    tasks = {
        "algorithms" : ["c2","c3","c4"],
        "datasets": ["ghisaconus"],
        "target_sizes" : list(range(30,1,-1))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=True, test=False)
    summary, details = ev.evaluate()
    plotter_classification.plot_combined(sources=["p6","p25"],only_algorithms=["bsnet","c1","c2","c3","c4"], only_datasets=["ghisaconus"])