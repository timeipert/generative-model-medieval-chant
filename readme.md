# Generative Models for Medieval Chants
<img width="677" height="86" alt="grafik" src="https://github.com/user-attachments/assets/ecdd8296-a792-405a-856b-7b143320b729" />

Overview:

* `models/`
  * `base_model.py` - Parent model
  * `model1.py` - Just a simple markov model of pitch strings
  * `model2.py` - mode parameter that changes transition matrix with a transpose() function
  * `model3.py` - mode parameter, where each mode has its own transition matrix
* `data/`
  * `data_gs.csv` - Graduale Synopticum Proper Chants ([View at Corpus monodicum](https://corpus-monodicum.de/search/http%3A%2F%2Fmonodicum%2Fdocument/?hideFromSearch=[%22http%3A%2F%2Fmonodicum%2FdocumentEditionsstatus%22]&hideFromTable=[%22http%3A%2F%2Fmonodicum%2FdocumentEditionsstatus%22]&query=[{%22uri%22%3A%22http%3A%2F%2Fmonodicum%2FdocumentEditionsstatus%22%2C%22value%22%3A%22zitiert%22}%2C{%22uri%22%3A%22http%3A%2F%2Fmonodicum%2FdocumentTextinitium%22%2C%22value%22%3A%22%22}%2C{%22uri%22%3A%22http%3A%2F%2Fmonodicum%2FdocumentGattung1%22%2C%22value%22%3A%22%22}%2C{%22uri%22%3A%22http%3A%2F%2Fmonodicum%2FdocumentFesttag%22%2C%22value%22%3A%22%22}%2C{%22uri%22%3A%22http%3A%2F%2Fmonodicum%2FdocumentQuelle_id%22%2C%22value%22%3A%22%22}]&pagination={%22asc%22%3Atrue%2C%22limit%22%3A10%2C%22offset%22%3A0%2C%22attribute%22%3A%22http%3A%2F%2Fmonodicum%2FdocumentDokumenten_idA%22}&navLinkHint=http%3A%2F%2Fmonodicum%2FdocumentZitiert))
  * `Introits.csv` - Corpus monodicum Introits ([View at Corpus monodicum](https://corpus-monodicum.de/search/http%3A%2F%2Fmonodicum%2Fdocument/?hideFromSearch=%5B%22http%3A%2F%2Fmonodicum%2FdocumentEditionsstatus%22%5D&hideFromTable=%5B%22http%3A%2F%2Fmonodicum%2FdocumentEditionsstatus%22%5D&query=%5B%7B%22uri%22%3A%22http%3A%2F%2Fmonodicum%2FdocumentEditionsstatus%22%2C%22value%22%3A%22ediert%22%7D%2C%7B%22uri%22%3A%22http%3A%2F%2Fmonodicum%2FdocumentTextinitium%22%2C%22value%22%3A%22%22%7D%2C%7B%22uri%22%3A%22http%3A%2F%2Fmonodicum%2FdocumentGattung1%22%2C%22value%22%3A%22Antiphon%22%7D%2C%7B%22uri%22%3A%22http%3A%2F%2Fmonodicum%2FdocumentFesttag%22%2C%22value%22%3A%22%22%7D%2C%7B%22uri%22%3A%22http%3A%2F%2Fmonodicum%2FdocumentQuelle_id%22%2C%22value%22%3A%22%22%7D%5D&pagination=%7B%22asc%22%3Atrue%2C%22limit%22%3A10%2C%22offset%22%3A0%2C%22attribute%22%3A%22http%3A%2F%2Fmonodicum%2FdocumentDokumenten_idA%22%7D&navLinkHint=http%3A%2F%2Fmonodicum%2FdocumentEdiert))
* `analysis/` - just a view helper functions for KL divergence, plotting, etc.
* `priors/` - methods for designing priors
* `samples/` - saved posterior samples
* `model[1-3].ipynb` - Jupyter notebooks for the models, *start here*
