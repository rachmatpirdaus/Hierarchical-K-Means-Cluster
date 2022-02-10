from mimetypes import init
from sklearn.metrics import silhouette_score
import warnings
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt  # MERENCANAKAN GRAFIK
import scipy.cluster.hierarchy as sch
import seaborn as sns
import pprint
from sklearn.preprocessing import StandardScaler
import multiprocessing
# get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
import seaborn as sns

from werkzeug.utils import secure_filename
import gower as dist
import seaborn as sns
import pprint
# get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.metrics import davies_bouldin_score
from sklearn_extra.cluster import KMedoids
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


#                           Library Flaskpy
#!venv/bin/python
import os
from flask import Flask, url_for, redirect, render_template, request, abort
from flask_sqlalchemy import SQLAlchemy
from flask_security import Security, SQLAlchemyUserDatastore, \
    UserMixin, RoleMixin, login_required, current_user
from flask_security.utils import encrypt_password
import flask_admin
from flask_admin.contrib import sqla
from flask_admin import helpers as admin_helpers
from flask_admin import BaseView, expose
from wtforms import PasswordField
import json
###############################################################################################################

# Create Flask application
app = Flask(__name__)
app.config.from_pyfile('config.py')
db = SQLAlchemy(app)
app.config['UPLOADED_PHOTOS'] = ''

# Define models
roles_users = db.Table(
    'roles_users',
    db.Column('user_id', db.Integer(), db.ForeignKey('user.id')),
    db.Column('role_id', db.Integer(), db.ForeignKey('role.id'))
)


class Role(db.Model, RoleMixin):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(80), unique=True)
    description = db.Column(db.String(255))

    def __str__(self):
        return self.name


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(255), nullable=False)
    last_name = db.Column(db.String(255))
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    active = db.Column(db.Boolean())
    confirmed_at = db.Column(db.DateTime())
    roles = db.relationship('Role', secondary=roles_users,
                            backref=db.backref('users', lazy='dynamic'))

    def __str__(self):
        return self.email


# Setup Flask-Security
user_datastore = SQLAlchemyUserDatastore(db, User, Role)
security = Security(app, user_datastore)


# Create customized model view class
class MyModelView(sqla.ModelView):

    def is_accessible(self):
        if not current_user.is_active or not current_user.is_authenticated:
            return False

        if current_user.has_role('superuser'):
            return True

        return False

    def _handle_view(self, name, **kwargs):
        """
        Override builtin _handle_view in order to redirect users when a view is not accessible.
        """
        if not self.is_accessible():
            if current_user.is_authenticated:
                # permission denied
                abort(403)
            else:
                # login
                return redirect(url_for('security.login', next=request.url))

    # can_edit = True
    edit_modal = True
    create_modal = True
    can_export = True
    can_view_details = True
    details_modal = True


class UserView(MyModelView):
    column_editable_list = ['email', 'first_name', 'last_name']
    column_searchable_list = column_editable_list
    column_exclude_list = ['password']
    #form_excluded_columns = column_exclude_list
    column_details_exclude_list = column_exclude_list
    column_filters = column_editable_list
    form_overrides = {
        'password': PasswordField
    }


class CustomView(BaseView):
    @expose('/')
    def index(self):
        return redirect(url_for('security.login', next=request.url))

class AgglomerativeClusteringView(BaseView):
    @expose('/',methods=('GET','POST'))
    def clustering(self):
        return self.render('admin/agglomerative_clustering_view.html', title='Hierarchical Clustering')
    
class KmeansClusteringView(BaseView):
    @expose('/')
    def kmeans_clustering(self):
        return self.render('admin/kmeans_clustering_view.html', title='K-Means Clustering')

# Flask views


@app.route('/')
def index():
    return redirect(url_for('security.login', next=request.url))

@app.route('/agglomerative_cluster/process',methods=("POST",))
def process_agglomerative():
    file1 = request.files['fileDesc']
    file2 = request.files['fileCek']
    filename = secure_filename(file1.filename)
    filename2 = secure_filename(file2.filename)
    file1.save(os.path.join(app.config['UPLOADED_PHOTOS'], filename))
    file2.save(os.path.join(app.config['UPLOADED_PHOTOS'], filename2))
    
    data = pd.read_csv('data_pahmi_labelled_baru.csv', sep=';', engine='python')
    data.astype('float')

    # # %% [markdown]
    # # Keterangan :
    # # `Sep` = merupakan parameter untuk menyatakan separator atau pemisah. Disini kita menggunakan pemisah koma.
    # # `Engine` = merupakan parameter untuk memaksa cell notebook menggunakan bahasa pemrograman yang diinginkan. Kasus disini menggunakan `Python`.
    # # %% [markdown]
    # # # **Preprocessing**
    # # Memilih kolom yang akan digunakan kedalam model agglomerative. Disini menggunakan fungsi `iloc` untuk mengambil kolom pertanyaan saja yang akan dimasukkan kedalam model dan mengubahnya menjadi array dengan perintah `.values`.

    # # %%
    x = data.iloc[:,1:-1].values
    x

    # # %% [markdown]
    # # # **Exploratory Data Analysis**
    # # Dilakukan EDA untuk melihat statistik deskriptif singkat dan visualisasi dendogram awal menggunakan berbagai macam linkage. Visualisasi ini hanyalah gambaran awal dari data yang sudah di lakukan pra proses.
    # # ## **Visualisasi dengan Linkage Ward**

    # # %%
    dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
    # plt.savefig('static/img/pahmi/dendogramward.png')
    # # # %% [markdown]
    # # # ## **Visualisasi dengan Linkage Average**

    # # # %%
    dendrogram = sch.dendrogram(sch.linkage(x, method='average'))
    # plt.savefig('static/img/pahmi/dendogramAverage.png')
    # # # %% [markdown]
    # # # ## **Visualisasi dengan Linkage Complete**

    # # # %%
    dendrogram = sch.dendrogram(sch.linkage(x, method='complete'))
    # plt.savefig('static/img/pahmi/dendogramComplete.png')
    # # %% [markdown]
    # # # **Data Modelling `Hierarchical Agglomerative`**
    # # 
    # # Disini menggunakan klaster berjumlah 3. Menggunakan distance metric `Euclidian Distance`. Selanjutnya menggunakan banyak jenis `linkage` untuk melihat karakteristik dendogram dan optimum cluster untuk data yang berhasil di load. Linkage yang digunakan diantaranya adalah :
    # # 1. Complete
    # # 2. Single
    # # 3. Average
    # # 4. Ward 

    # # %%
    model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
    model.fit(x)
    labels = model.labels_

    # # %% [markdown]
    # # ## **Visualisasi Sebaran data yang sudah diimplementasikan algoritma**
    # # Parameter :
    # # * s = menyatakan ukuran scatter
    # # * marker = digunakan 'o' untuk menampilkan sebaran data dengan simbol bulat kecil
    # # * color = warna data

    # # %
    plt.scatter(x[labels==0, 0], x[labels==0, 1], s=25, marker='o', color='purple')
    plt.scatter(x[labels==1, 0], x[labels==1, 1], s=25, marker='o', color='blue')
    plt.scatter(x[labels==2, 0], x[labels==2, 1], s=25, marker='o', color='green')
  
   

    # # plt.show()

    # # %% [markdown]
    # # # **Tampilan Data Responden yang telah memiliki labels**

    # # %%
    labeled_data = pd.read_csv('data_pahmi_labelled_baru.csv', sep=';', engine='python')
    labeled_data


    # # %%
    model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    model.fit(x)
    labels = model.labels_
    plt.scatter(x[labels==0, 0], x[labels==0, 1], s=25, marker='o', color='purple')
    plt.scatter(x[labels==1, 0], x[labels==1, 1], s=25, marker='o', color='blue')
    plt.scatter(x[labels==2, 0], x[labels==2, 1], s=25, marker='o', color='green')
    
    # # plt.show()


    # # %%
    labeled_described = labeled_data.describe()
    # # %%
    pd.Series(labels).value_counts()
    # # %%
    data_scaled = normalize(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    data_scaled.head()

    # # %% [markdown]
    # # # **Melakukan Prediksi**

    # # %%
    y = model.fit_predict(x)

    # # %% [markdown]
    # # # **Visualisasi Cluster dengan Dendogram**

    # # %%
    plt.figure(figsize=(10,7))
    dendrogram = sch.dendrogram(sch.linkage(x, method = 'complete'))
    plt.title('Clustering')
    plt.xlabel('Posbindu')
    plt.ylabel('jarak euclidean')
    # # plt.savefig('static/img/visualisasiclusterdendogram.png')
    # # plt.show()

    # # %%
    model = AgglomerativeClustering(n_clusters=3, affinity='euclidean',
    linkage='ward')
    model.fit(x)
    labels = model.labels_

    # # # **Validasi Menggunakan Davies Bouldin Score**
    # # %%
    davies_bouldin_score(x, labels)
    # # %%
    range_n_clusters = range(2,7)
    cluster_DBI=[]
    for n_clusters in range_n_clusters:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters )
        preds = clusterer.fit_predict(x)
        score = davies_bouldin_score(x, preds )
        cluster_DBI.append(score)
    plt.figure(figsize=(10,6))
    plt.title('DBI Curve for Finding Optimum K value')
    plt.xlabel('No. of clusters')
    plt.ylabel('DBI Score')
    plt.plot(list(range_n_clusters),cluster_DBI,marker='o')
    # # plt.show()
    # # plt.savefig('static/img/kurvaDBI.png')


    # # check
    data_csv = pd.DataFrame(data).to_dict('records')
    data_labelled = pd.DataFrame(labeled_data).to_dict('records')
    data_scaled = data_scaled.to_dict('records')
    labelled_describe = pd.DataFrame(labeled_described).to_dict('records')

    return dict(
        data_csv1 = data_csv,
        data_csv2 = data_labelled,
        data_scaled=data_scaled,
        labeled_describe=labelled_describe
    )


@app.route('/kmeans_cluster/process',methods=("POST",))
def process_kmeans():
        
    file1 = request.files['fileDesc']
    filename = secure_filename(file1.filename)
    file1.save(os.path.join(app.config['UPLOADED_PHOTOS'], filename))
    # %%
    data =pd.read_csv('data_pahmi_labelled_baru.csv', sep=';', engine='python')
    data.astype('float')

    # %% [markdown]
    # Keterangan :
    # `Sep` = merupakan parameter untuk menyatakan separator atau pemisah. Disini kita menggunakan pemisah koma.
    # `Engine` = merupakan parameter untuk memaksa cell notebook menggunakan bahasa pemrograman yang diinginkan. Kasus disini menggunakan `Python`.
    # %% [markdown]
    # # **Normalisasi**
    # # %%
    labeled_described = data.describe()
    # %%
    from sklearn.preprocessing import normalize
    data_scaled = normalize(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    data_scaled.head()


    # %%
    # **Preprocessing** 
    #Memilih kolom yang akandigunakan kedalam model K-means. Disini menggunakan fungsi `iloc` untuk mengambil kolom pertanyaan saja yang akan dimasukkan kedalam model dan mengubahnya menjadi array dengan perintah `.values`.


    # %%
    x = data_scaled.iloc[:,1:-1].values
    x

    # %% [markdown]
    # # **Data Modelling `K-Means`**

    # %%
    import warnings
    from sklearn.cluster import KMeans
    warnings.filterwarnings("ignore")
    model = KMeans(n_clusters=3, n_init=12, random_state=0)
    model.fit(x)
    labels = model.labels_
    # %%
    pd.Series(labels).value_counts()
    # %%
    km4=KMeans(n_clusters=3, max_iter=300, n_init=10, random_state=0)
    y_means = km4.fit_predict(data)
    labels = km4.labels_
    data['clusters'] = km4.predict(data)
    # %%
    pd.Series(labels).value_counts()

    data.describe()
    # %%
    from sklearn import datasets

    # %%
    data_std=data[['Posbindu', 'Umur', 'Jenis Kelamin', 'Merokok', 'Sistol', 'IMT']]
    continous_vars=data_std.describe().columns
    data_std.hist(column=continous_vars,figsize=(16,16))
    # %%
    from sklearn.cluster import KMeans
    wcss = []
    for i in range (1,11):
        km=KMeans(n_clusters=i, max_iter=300, n_init=10, random_state=0)
        km.fit(data)
        wcss.append(km.inertia_)
    # %%
    km4=KMeans(n_clusters=3, max_iter=300, n_init=10, random_state=0)
    y_means= km4.fit_predict(data)
    labels=km4.labels_
    plt.scatter(x[labels==0, 0], x[labels==0, 1], s=25, marker='o', color='purple')
    plt.scatter(x[labels==1, 0], x[labels==1, 1], s=25, marker='o', color='blue')
    plt.scatter(x[labels==2, 0], x[labels==2, 1], s=25, marker='o', color='green')
  

    # %%
    km4=KMeans(n_clusters=3, max_iter=300, n_init=10, random_state=0)
    y_means= km4.fit_predict(data)
    labels=km4.labels_
    davies_bouldin_score(x, labels)
    range_n_clusters = range(2,5)
    cluster_DBI=[]
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state =None)
        preds = clusterer.fit_predict(x)
        score = davies_bouldin_score(x, preds)
        cluster_DBI.append(score)
    # plt.figure(figsize=[10,6])
    # plt.title('DBI Curve for Finding Optimum K value')
    # plt.xlabel('No. of clusters')
    # plt.ylabel('DBI Score')
    # plt.plot(list(range_n_clusters),cluster_DBI,marker='o')
    # plt.savefig('static/img/kmedoid1.png')


    # %%
    data.isnull().any()


    # %%
    # # X, Y = make_blobs(n_samples=136, random_state=20)


    # %%
   # # sns.scatterplot(X[:,0],X[:,1], hue = Y)
   # # plt.savefig('static/img/scatter_kmedoid1.png')

    # %%
   # # df = pd.DataFrame(x)
   # # ms = MeanShift()
   # # ms.fit(df)


    # %%
    set(labels)


    # %%
   # # sns.scatterplot(X[:,0],X[:,1], hue=labels)
   # # plt.savefig('static/img/scatter_kmedoid2.png')

    # %%
   


    # %%
   # # silhouette_score(X,Y) , silhouette_score,(X, ms.labels_)

    data_csv = pd.DataFrame(data).to_dict('records')
    data_scaled = data_scaled.to_dict('records')
    return dict(
        data_csv = data_csv,
        data_scaled = data_scaled
    )


# Create admin
admin = flask_admin.Admin(
    app,
    'My Dashboard',
    base_template='my_master.html',
    template_mode='bootstrap4',
)

# Add model views
admin.add_view(AgglomerativeClusteringView(name="Agglomerative Clustering",
               menu_icon_type='fa', menu_icon_value='fa-bar-chart',))
admin.add_view(KmeansClusteringView(name="Kmeans Clustering",
               menu_icon_type='fa', menu_icon_value='fa-bar-chart',))


# define a context processor for merging flask-admin's template context into the
# flask-security views.


@security.context_processor
def security_context_processor():
    return dict(
        admin_base_template=admin.base_template,
        admin_view=admin.index_view,
        h=admin_helpers,
        get_url=url_for
    )


def build_sample_db():
    """
    Populate a small db with some example entries.
    """

    import string
    import random

    db.drop_all()
    db.create_all()

    with app.app_context():
        user_role = Role(name='user')
        super_user_role = Role(name='superuser')
        db.session.add(user_role)
        db.session.add(super_user_role)
        db.session.commit()

        test_user = user_datastore.create_user(
            first_name='Admin',
            email='admin',
            password=encrypt_password('admin'),
            roles=[user_role, super_user_role]
        )

        first_names = [
            'Harry', 'Amelia', 'Oliver', 'Jack', 'Isabella', 'Charlie', 'Sophie', 'Mia',
            'Jacob', 'Thomas', 'Emily', 'Lily', 'Ava', 'Isla', 'Alfie', 'Olivia', 'Jessica',
            'Riley', 'William', 'James', 'Geoffrey', 'Lisa', 'Benjamin', 'Stacey', 'Lucy'
        ]
        last_names = [
            'Brown', 'Smith', 'Patel', 'Jones', 'Williams', 'Johnson', 'Taylor', 'Thomas',
            'Roberts', 'Khan', 'Lewis', 'Jackson', 'Clarke', 'James', 'Phillips', 'Wilson',
            'Ali', 'Mason', 'Mitchell', 'Rose', 'Davis', 'Davies', 'Rodriguez', 'Cox', 'Alexander'
        ]

        for i in range(len(first_names)):
            tmp_email = first_names[i].lower() + "." + \
                last_names[i].lower() + "@example.com"
            tmp_pass = ''.join(random.choice(
                string.ascii_lowercase + string.digits) for i in range(10))
            user_datastore.create_user(
                first_name=first_names[i],
                last_name=last_names[i],
                email=tmp_email,
                password=encrypt_password(tmp_pass),
                roles=[user_role, ]
            )
        db.session.commit()
        
    return


if __name__ == '__main__':

    # Build a sample db on the fly, if one does not exist yet.
    app_dir = os.path.realpath(os.path.dirname(__file__))
    database_path = os.path.join(app_dir, app.config['DATABASE_FILE'])
    if not os.path.exists(database_path):
        build_sample_db()

    # Start app
    app.run(debug=True)


