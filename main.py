
import os
print('pid:', os.getpid())

from time import time
from par_parser import get_args
from chem_lib.models import ContextAwareRelationNet, Meta_Trainer
from chem_lib.utils import count_model_params


def main():
    root_dir = '.'
    args = get_args(root_dir)

    model = ContextAwareRelationNet(args)
    count_model_params(model)
    model = model.to(args.device)
    trainer = Meta_Trainer(args, model)

    t1=time()
    print('Initial Evaluation')
    best_avg_auc=0
    for epoch in range(1, args.epochs + 1):
        print('----------------- Epoch:', epoch,' -----------------')
        trainer.train_step()

        if epoch % args.eval_steps == 0 or epoch==1 or epoch==args.epochs:
            print('Evaluation on epoch',epoch)
            best_avg_auc = trainer.test_step()

        if epoch % args.save_steps == 0:
            trainer.save_model()
        print('Time cost (min):', round((time()-t1)/60,3))
        t1=time()

    print('Train done.')
    print('Best Avg AUC:',best_avg_auc)

    trainer.conclude()

    if args.save_logs:
        trainer.save_result_log()

if __name__ == "__main__":
    main()

