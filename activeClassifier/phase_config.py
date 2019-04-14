def get_phases(FLAGS):
    return [
            {'name'      : 'Pre',
             'num_epochs': FLAGS.pre_train_epochs,
             'policy'    : FLAGS.pre_train_policy if (FLAGS.pre_train_policy != 'same') else FLAGS.planner,
             'incl_uk'   : FLAGS.pre_train_uk,
             'final_eval': False},
            {'name'      : 'Full',
             'num_epochs': FLAGS.num_epochs - FLAGS.pre_train_epochs,
             'policy'    : FLAGS.planner,
             'incl_uk'   : True,
             'final_eval': True},
          ]