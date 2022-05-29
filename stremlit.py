import streamlit as st
from sklearn.datasets import load_wine
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from streamlit_option_menu import option_menu
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from streamlit.legacy_caching.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
import numpy as np
import pandas as pd

selected2 = option_menu("Welcome to my first app", [ 'Make yourself at home'], 
    icons=['house'], 
    menu_icon="cast", default_index=0, orientation="horizontal")



class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


st_state = _get_state()



st.write("""
### Now you will be able to train a Decision Tree Classifier""")

st.write("""
#### You need to choose your parameters""")

criterion = st.selectbox(
     'Choose criterion (the function to measure the quality of a split)',
     ('gini', 'entropy', 'log_loss'))

splitter = st.selectbox(
     'Choose splitter (The strategy used to choose the split at each node)',
     ('best', 'random'))

max_depth = st.number_input('Choose max_depth of the tree', min_value=0, format ='%d') #default None



data = load_wine(as_frame=True)
wine_info = data.data
wine_info['target'] = data.target


norm_data = preprocessing.normalize(wine_info.loc[:, wine_info.columns!='target'])
target = wine_info.target
X_train, X_test, y_train, y_test = train_test_split(norm_data, target, test_size=0.2, random_state=42)



if st.button('Train model and get training results') or st_state.button_run:
    st_state.button_run = True

    if max_depth!=0:
        decis_tree_model = DecisionTreeClassifier(criterion=criterion,splitter=splitter,max_depth=max_depth)
    else:
        decis_tree_model = DecisionTreeClassifier(criterion=criterion,splitter=splitter)

    f1 = cross_val_score(decis_tree_model, X_train, y_train, cv=5, scoring='f1_weighted').mean()
    balanced_accuracy = cross_val_score(decis_tree_model, X_train, y_train, cv=5, scoring='balanced_accuracy').mean()
    st.write( '### Результат модели решающего дерева:')
    st.write('f1_weighted: '+ str(round(f1,2)) + ', balanced_accuracy: ' + str(round(balanced_accuracy,2)))

    st.write('--------------------------------------------------------------------------')
    if st.button('Control model on test data'):
    
        if max_depth!=0:
            decis_tree_model = DecisionTreeClassifier(criterion=criterion,splitter=splitter,max_depth=max_depth)
        else:
            decis_tree_model = DecisionTreeClassifier(criterion=criterion,splitter=splitter)
    
        decis_tree_model.fit(X_train, y_train)
        svm_res = decis_tree_model.predict(X_test)
        bascore = balanced_accuracy_score(y_test, svm_res)
        f1 = f1_score(y_test, svm_res, average='weighted')
        st.write( '### Result of decision tree model:')
        st.write('f1_weighted: '+ str(round(f1,2)) + ', balanced_accuracy: ' + str(round(bascore,2)))

        st.write( '*Check, in which classes classifier has more errors:*')
        errors_true = [val for num,val in enumerate(y_test) if val!=svm_res[num]]
        errors_pred = [svm_res[num] for num,val in enumerate(y_test) if val!=svm_res[num]]
        st.bar_chart(errors_true)

        errs = pd.DataFrame(data=np.zeros((3,4), dtype='int'),columns=["true\pred","0", "1", "2"])
        errs['true\pred'] = ('-','-','-')
        for i in range(len(errors_true)):
            errs[str(errors_pred[i])][errors_true[i]]+=1
        st.dataframe(errs)