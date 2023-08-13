from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
from ignite.engine import Events, Engine
from ignite.metrics import Average, Loss
from ignite.contrib.handlers import ProgressBar
import gpytorch
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood
from due.dkl import DKL, GP, initial_values
from due.fc_resnet import FCResNet


def basic_due(
    dataset,
    x_field="emb_smiles",
    y_field="pic50",
    save_as="due_model.pkl",
    load_as=None,
    continue_training=False,
    steps=1e5,
    depth=4,
    batch_size=512,
    remove_spectral_norm=False,
    test_frac=0.03,
    random_seed=510,
):
    """
    Train a basic DUE model on a dataset.

    The DUE model is based on the following paper:

    On Feature Collapse and Deep Kernel Learning for Single Forward Pass Uncertainty
    https://arxiv.org/abs/2102.11409

    E.g., see Algorithm 1 and https://github.com/y0ast/DUE.

    Returns:
        Model: The trained DUE model.
        Regressed results: The regressed results on the test set with
          uncertainties.
    """
    np.random.seed(seed=random_seed)

    x = np.stack([r[x_field] for r in dataset], 0)
    if type(y_field) == str:
        y = np.stack([r[y_field] for r in dataset], 0)
    elif type(y_field) == list:
        y = np.stack(
            [np.stack([r[y_fieldi] for y_fieldi in y_field], -1) for r in dataset], 0
        )

    perm = np.random.permutation(len(dataset))
    test_rec_indices = perm[: int(test_frac * len(dataset))]
    train_rec_indices = perm[int(test_frac * len(dataset)) :]

    train_x = torch.tensor(x[train_rec_indices], dtype=torch.float)
    train_y = torch.tensor(y[train_rec_indices], dtype=torch.float)
    test_x = torch.tensor(x[test_rec_indices], dtype=torch.float)
    test_y = torch.tensor(y[test_rec_indices], dtype=torch.float)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    n_samples = train_x.shape[0]
    epochs = steps // len(train_loader) + 1

    input_dim = train_x.shape[-1]
    features = 256
    num_outputs = 1
    spectral_normalization = True
    coeff = 0.95
    n_inducing_points = 60
    n_power_iterations = 2
    dropout_rate = 0.03

    # ResFNN architecture
    feature_extractor = FCResNet(
        input_dim=input_dim,
        features=features,
        depth=depth,
        spectral_normalization=spectral_normalization,
        coeff=coeff,
        n_power_iterations=n_power_iterations,
        dropout_rate=dropout_rate,
    )
    kernel = "RBF"
    initial_inducing_points, initial_lengthscale = initial_values(
        train_dataset, feature_extractor, n_inducing_points
    )

    # Gaussian process (GP)
    gp = GP(
        num_outputs=num_outputs,
        initial_lengthscale=initial_lengthscale,
        initial_inducing_points=initial_inducing_points,
        kernel=kernel,
    )

    # Deep Kernel Learning (DKL) model
    model = DKL(feature_extractor, gp)

    likelihood = GaussianLikelihood()
    elbo_fn = VariationalELBO(likelihood, model.gp, num_data=len(train_dataset))
    loss_fn = lambda x, y: -elbo_fn(x, y)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    lr = 1e-3
    parameters = [
        {"params": model.parameters(), "lr": lr},
    ]
    parameters.append({"params": likelihood.parameters(), "lr": lr})
    optimizer = torch.optim.Adam(parameters)

    def step(engine, batch):
        model.train()
        likelihood.train()
        optimizer.zero_grad()

        x, y = batch
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_step(engine, batch):
        model.eval()
        likelihood.eval()
        x, y = batch
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        y_pred = model(x)
        return y_pred, y

    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Average()
    metric.attach(trainer, "loss")
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer)

    metric = Loss(lambda y_pred, y: -likelihood.expected_log_prob(y, y_pred).mean())
    metric.attach(evaluator, "loss")

    if not load_as is None:
        read = torch.load(load_as)
        model.load_state_dict(read)

    if load_as is None or continue_training:
        print(f"Training with {n_samples} datapoints for {epochs} epochs")

        @trainer.on(Events.EPOCH_COMPLETED(every=int(epochs / 10) + 1))
        def log_results(trainer):
            evaluator.run(test_loader)
            print(
                f"Results - Epoch: {trainer.state.epoch} - "
                f"Test Likelihood: {evaluator.state.metrics['loss']:.2f} - "
                f"Loss: {trainer.state.metrics['loss']:.2f}"
            )

        trainer.run(train_loader, max_epochs=epochs)
        model.eval()
        likelihood.eval()
        torch.save(model.state_dict(), save_as)

    # If you want to differentiate the model.
    if remove_spectral_norm:
        model.feature_extractor.first = torch.nn.utils.remove_spectral_norm(
            model.feature_extractor.first
        )

    Xs_, Ys_, dYs_ = [], [], []
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(64):
        for batch_x, batch_y in test_loader:
            pred = model(batch_x.cuda())
            mean = pred.mean.cpu().numpy()
            std = pred.stddev.cpu().numpy()
            Xs_.append(batch_y.detach().cpu().numpy())
            Ys_.append(mean)
            dYs_.append(std)

    Xs = np.concatenate(Xs_, 0)
    Ys = np.concatenate(Ys_, 0)
    dYs = np.concatenate(dYs_, 0)

    return model, (Xs, Ys, dYs)
