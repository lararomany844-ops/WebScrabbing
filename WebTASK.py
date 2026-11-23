import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


def main():
  # ==========================================
  # 1. SCRAPING 
  # ==========================================
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
          " (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
      )
  }

  # --- (Source 1) ---
  url_1 = "https://webscraper.io/test-sites/e-commerce/allinone/phones/touch"
  response_1 = requests.get(url_1, headers=headers)
  soup_1 = BeautifulSoup(response_1.text, "html.parser")

  data_source_1 = []
  for item in soup_1.find_all("div", class_="thumbnail"):
    title_tag = item.find("a", class_="title")
    price_tag = item.find("h4", class_="price")
    rating_tag = item.find("p", attrs={"data-rating": True})

    title = title_tag.text.strip() if title_tag else np.nan
    price = price_tag.text.strip() if price_tag else np.nan
    rating = f"{rating_tag['data-rating']} out of 5" if rating_tag else np.nan

    data_source_1.append({
        "Product_Name": title,
        "Price": price,
        "Rating": rating,
        "Source": "TechStore_A",
    })

  # --- (Source 2) ---
  url_2 = (
      "https://books.toscrape.com/catalogue/category/books/science_22/index.html"
  )
  response_2 = requests.get(url_2, headers=headers)
  soup_2 = BeautifulSoup(response_2.text, "html.parser")

  rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
  data_source_2 = []

  for item in soup_2.find_all("article", class_="product_pod"):
    title_tag = item.find("h3").find("a") if item.find("h3") else None
    price_tag = item.find("p", class_="price_color")
    rating_el = item.find("p", class_="star-rating")

    title = title_tag["title"] if title_tag and title_tag.has_attr("title") else np.nan
    raw_price = price_tag.text.strip() if price_tag else ""
    price = raw_price.replace("Â£", "").replace("£", "").strip() + " USD" if raw_price else np.nan

    if rating_el and len(rating_el.get("class", [])) > 1:
      rating_num = rating_map.get(rating_el["class"][1], np.nan)
      rating = f"{rating_num} / 5" if pd.notna(rating_num) else np.nan
    else:
      rating = np.nan

    data_source_2.append({
        "Product_Name": f"  {title}  " if pd.notna(title) else np.nan,
        "Price": price,
        "Rating": rating,
        "Source": "GlobalMart_B",
    })

  # ==========================================
  # 2. DATA COMBINATION 
  # ==========================================
  df1 = pd.DataFrame(data_source_1)
  df2 = pd.DataFrame(data_source_2)

  combined_df = pd.concat([df1, df2], ignore_index=True)

  if len(combined_df) > 0:
    combined_df = pd.concat([combined_df, combined_df.iloc[[0]]], ignore_index=True)
    combined_df.loc[min(2, len(combined_df) - 1), "Rating"] = np.nan

  # ==========================================
  #(BEFORE PREPROCESSING)
  # ==========================================
  print("================ BEFORE PREPROCESSING ================")
  print("\n--- Head ---")
  print(combined_df.head())

  print("\n--- Info ---")
  print(combined_df.info())

  print("\n--- Describe ---")
  print(combined_df.describe(include="all"))

  print("\n--- Missing Values (isnull().sum()) ---")
  print(combined_df.isnull().sum())

  # ==========================================
  # 3. DATA PREPROCESSING 
  # ==========================================
  clean_df = combined_df.copy()

  # Text Cleaning: 
  clean_df["Product_Name"] = (
      clean_df["Product_Name"].astype(str).str.strip().str.lower()
  )
  clean_df["Product_Name"] = clean_df["Product_Name"].replace("nan", np.nan)
  clean_df["Source"] = clean_df["Source"].astype(str).str.strip()

  # Data Type Conversion: 
  clean_df["Price"] = (
      clean_df["Price"]
      .astype(str)
      .str.replace(r"[^0-9.]", "", regex=True)
      .replace("", np.nan)
      .astype(float)
  )
  clean_df["Rating"] = (
      clean_df["Rating"]
      .astype(str)
      .str.extract(r"(\d+\.?\d*)")[0]
      .astype(float)
  )

  # Handling Missing Values:
  mean_rating = round(clean_df["Rating"].mean(), 1)
  clean_df["Rating"] = clean_df["Rating"].fillna(mean_rating)
  clean_df["Product_Name"] = clean_df["Product_Name"].fillna("unknown product")

  # Deduplication: 
  clean_df = clean_df.drop_duplicates().reset_index(drop=True)

  # ==========================================
  #(AFTER PREPROCESSING)
  # ==========================================
  print("\n================ AFTER PREPROCESSING ================")
  print("\n--- Head ---")
  print(clean_df.head())

  print("\n--- Info ---")
  print(clean_df.info())

  print("\n--- Describe ---")
  print(clean_df.describe())

  print("\n--- Missing Values (isnull().sum()) ---")
  print(clean_df.isnull().sum())

  # CSV
  clean_df.to_csv("cleaned_dataset.csv", index=False)
  print("\n : cleaned_dataset.csv")


if __name__ == "__main__":
  main()
