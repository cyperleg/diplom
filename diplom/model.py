import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from diplom.data import CNNDataset
from tqdm import tqdm
import plotly.graph_objs as go
from diplom.functions import get_model_summary, generate_spectre


# device setting cuda or cpu
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# generate data
gen_data = CNNDataset(20000, device, load=True)
test_data = generate_spectre()

train_data, val_data = random_split(gen_data, (0.8, 0.2))

# batching data
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# model describing
model = nn.Sequential(
    nn.Conv2d(1, 8, (5, 5), padding=2),
    nn.ReLU(True),
    nn.MaxPool2d(4),
    nn.Conv2d(8, 16, (3, 3), padding=1),
    nn.ReLU(True),
    nn.MaxPool2d(4),
    nn.Conv2d(16, 32, (3, 3), padding=1),
    nn.ReLU(True),
    nn.MaxPool2d(4),
    nn.Flatten(),
    nn.Linear(32 * 4 * 4, 16 * 4 * 4),
    nn.ReLU(True),
    nn.Linear(16 * 4 * 4, 16 * 2 * 2),
    nn.ReLU(True),
    nn.Linear(64, 32),
    nn.ReLU(True),
    nn.Linear(32, 5)
).to(device)

# optimize and loss function
loss_classification = nn.MSELoss()
opt_classification = optim.Adam(model.parameters(), lr=0.001)

# global vars for metrics
train_loss = []
val_loss = []

EPOCH = 30

for epoch in range(EPOCH):
    model.train()

    running_training_loss = []
    true_answer = 0

    # console better UI
    train_loop = tqdm(train_loader, leave=False)

    # training loop
    for x, targets in train_loop:
        # data setting
        x = x.to(device)
        targets = targets.to(device)

        # getting prediction and loss
        pred = model(x)
        loss = loss_classification(pred, targets)

        # back propogation
        opt_classification.zero_grad()
        loss.backward()
        opt_classification.step()

        # running metrics
        running_training_loss.append(loss.item())
        mean_train_loss = sum(running_training_loss) / len(running_training_loss)

        train_loop.set_description(f"Epoch [{epoch + 1} / {EPOCH}], train_loss={mean_train_loss:.4f}")

    # epoch metrics
    train_loss.append(mean_train_loss)

    # validation loop
    model.eval()
    with torch.no_grad():
        running_val_loss = []
        true_answer = 0

        for x, targets in val_loader:
            x = x.to(device)
            targets = targets.to(device)

            pred = model(x)
            loss = loss_classification(pred, targets)

            running_val_loss.append(loss.item())
            mean_val_loss = sum(running_val_loss) / len(running_val_loss)

    val_loss.append(mean_val_loss)

    train_loop.set_description(
        f"Epoch [{epoch + 1} / {EPOCH}], train_loss={mean_train_loss:.4f}, val_loss={mean_val_loss:.4f}")

# graph of metrics
fig_1 = go.Figure()
fig_1.add_trace(go.Scatter(y=train_loss, name="train_loss", mode="lines"))
fig_1.add_trace(go.Scatter(y=val_loss, name="val_loss", mode="lines"))

text_to_display = f"Validation loss = {str(val_loss[-1])} Train loss = {str(train_loss[-1])} <br>"

res = []

with torch.no_grad():
    pred = model(torch.tensor(test_data[0], device=device, dtype=torch.float32).unsqueeze_(0))
    target = torch.tensor(test_data[1], device=device, dtype=torch.float32)
    res.append(pred)
    res.append(target)
    res.append(loss_classification(pred, target))

res_text = f"<br>pred = {res[0]}, target = {res[1]}, loss = {res[2]}"

fig_1.update_layout(
    annotations=[
        go.layout.Annotation(
            text=text_to_display + get_model_summary(model, input_size=(1, 256, 256)).replace('\n', '<br>') + res_text,
            xref="paper", yref="paper",
            x=0.5, y=-1.35,
            showarrow=False,
            font=dict(size=12)
        )
    ],
    margin=dict(b=600),
    height=1100
)

fig_1.show()
