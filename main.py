import time
from plotly.offline import init_notebook_mode, iplot
from utils.preprocess import crop_imgs, load_data, preprocess_data, preprocess_imgs, save_new_images, init_train_data, train_data_generator
from utils.trainer import setup_train

init_notebook_mode(connected=True)

if __name__ == "__main__":
    init_train_data()
    X_val_prep, X_test_prep, y_val, y_test = preprocess_data()
    train_generator, validation_generator = train_data_generator()[:2]
    trainer = setup_train()

    start = time.time()

    vgg16_history = trainer.fit_generator(
        train_generator,
        # steps_per_epoch=20,
        epochs=100,
        validation_data=validation_generator,
        # validation_steps=30,
    )

    end = time.time()
    print(end - start)

    # validate on val set
    predictions = trainer.predict(X_test_prep)
    predictions = [1 if x > 0.5 else 0 for x in predictions]

    _, train_acc = trainer.evaluate(X_val_prep, y_val, verbose=0)
    _, test_acc = trainer.evaluate(X_test_prep, y_test, verbose=0)
