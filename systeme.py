# importer les bibliothèques
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import streamlit as st # type: ignore

# Chargement des données
st.cache(hash_funcs={pd.DataFrame: lambda _: None})
def load_data():
    data = {
        'name': ['Mione K1 ROM 3GB RAM 32GB 5.99 Inch display 4G...',
                 'Porsche design Huawei Mate 20 RS Dual SIM - 512GB',
                 'Samsung Galaxy Fold Dual SIM - 512GB',
                 'Samsung Galaxy Fold Dual SIM - 512GB',
                 'Blackberry P\'9981 Porsche Design - 8GB',
                 'Huawei P40 Pro Single SIM and E-SIM - 256GB',
                 'Apple iPhone XR without Face Time - 64GB',
                 'Samsung Galaxy S10 Plus Dual Sim - 1Tb',
                 'Apple iPhone 11 with FaceTime - 256GB',
                 'Samsung Galaxy S20 Plus Dual SIM - 128GB'],
        'item_price': ['227,599.00 SAR', '8,500.00 SAR', '7,587.00 SAR', '7,099.00 SAR',
                       '6,550.00 SAR', '3,499.00 SAR', '3,498.00 SAR', '3,489.00 SAR',
                       '3,487.97 SAR', '3,398.98 SAR'],
        'item_brand_name': ['mione', 'huawei', 'samsung', 'samsung', 'blackberry',
                            'huawei', 'apple', 'samsung', 'apple', 'samsung']
    }
    return pd.DataFrame(data)

# Fonction de recommandation
def recommend_similar_products(product_name, df):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['name'])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    product_index = df[df['name'] == product_name].index[0]
    similar_products = similarity_matrix[product_index].argsort()[:-1][::-1]

    top_similar_products = []
    for idx in similar_products[:4]:
        top_similar_products.append(df.loc[idx, 'name'])

    return top_similar_products

# Application Streamlit
def main():
 background_image = "C:/Users/Livine/Desktop/image.jpg"
 st.image(background_image, use_column_width=True)
    

# Afficher l'image en arrière-plan
 st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url("{background_image}") no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.title('SYSTEME DE RECOMMAMDATION DES TELEPHONES')

    # Chargement des données
df = load_data()

    # Sélection du produit à recommander
product_name = st.selectbox('Sélectionnez un produit :', df['name'])

    # Affichage des produits recommandés
st.subheader('Produits Recommandés :')
recommendations = recommend_similar_products(product_name, df)
for product in recommendations:
        st.write("- " + product)

if __name__ == '__main__':
    main()

