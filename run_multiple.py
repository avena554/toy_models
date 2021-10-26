from torch.optim import Adam
import argparse
import os

from text_classif_tests import run_one, preprocess, make_dataloader, build_model, parser, LinearModelF1,\
    SEED, DEVICE


def run_multi(train, dev, test, voc, df_oc, embeddings, n_docs_train, configs, times, measure, store_action, device,
              optimizer_type=Adam):
    for i, (config, n) in enumerate(zip(configs, times)):
        train_dl, dev_dl, test_dl = make_dataloader(voc, train, dev, test, config)
        model, loss = build_model(config, voc, df_oc, device, n_docs_train, embeddings=embeddings)
        for j in range(n):
            print("running {:s}...".format(config.model_name))
            run_one(train_dl, dev_dl, test_dl, model, loss, measure, device, config, optimizer_type)
            store_action(i + 1, j + 1, measure.value())
            print("...done")


if __name__ == "__main__":
    m_parser = argparse.ArgumentParser(description='train multiple BOW model on the IMDB dataset')
    m_parser.add_argument("--i", dest="input_file", required=True)
    m_parser.add_argument("--o", dest="output_file", required=True)
    m_parser.add_argument("--truncate", dest="truncate", action="store_true", help="truncate corpus", default=False)
    m_parser.add_argument("--trunc-size", dest="trunc_size", type=int, help="size of truncated corpus",
                          default=100)
    m_parser.add_argument("--dev-prop", dest="dev_prop", help="dev corpus size in proportion w.r.t. train",
                          default=0.25)
    m_parser.add_argument("--pretrained-we", dest="we_file", help="path to pretrained word-embeddings", default=None)
    m_parser.add_argument("--h-size", dest="h_size", type=int, help="size of hidden layer(s)", default=200)

    args = m_parser.parse_args()

    try:
        os.remove(args.output_file)
    except FileNotFoundError:
        pass

    def write_res(exp_id, num_it, value):
        with open(args.output_file, "a") as out_f:
            if num_it == 1 and exp_id != 1:
                out_f.write("\n")
            else:
                if num_it > 1:
                    out_f.write("\t")

            out_f.write(str(value))

    with open(args.input_file, 'r') as in_f:
        m_times = []
        m_configs = []
        lines = in_f.readlines()
        in_iterator = iter(lines)
        try:
            while in_iterator:
                params = next(in_iterator)
                m_num_it_str = next(in_iterator)
                exp_config = parser.parse_args(params.split(" "))
                m_num_it = int(m_num_it_str)
                m_configs.append(exp_config)
                m_times.append(m_num_it)
        except StopIteration:
            pass

    m_train, m_dev, m_test, m_voc, m_df_oc, m_gf, m_embeddings = preprocess(args, SEED)
    print(f"vocabulary size: {len(m_voc):d}")
    m_measure = LinearModelF1(DEVICE)
    run_multi(m_train, m_dev, m_test, m_voc, m_df_oc, m_embeddings, len(m_train), m_configs, m_times, m_measure, write_res, DEVICE)
