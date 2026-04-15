import pandas as pd

# ============================================
# 1️⃣ LOAD & CLEAN PHENOTYPE
# ============================================

pheno = pd.read_csv("dataset/BLUPS_PH_EH_.txt", sep="\t")
print("Original phenotype shape:", pheno.shape)

# Remove NA phenotype
pheno_clean = pheno.dropna(subset=["PH", "EH"])
print("After removing NA:", pheno_clean.shape)

# Remove popcorn lines
popcorn = pd.read_csv("dataset/popcorn.txt", header=None)
popcorn_list = popcorn[0].tolist()

pheno_clean = pheno_clean[~pheno_clean["Full_name"].isin(popcorn_list)]
print("After removing popcorn lines:", pheno_clean.shape)

# Extract Base_ID
pheno_clean["Base_ID"] = pheno_clean["Full_name"].str.split(".").str[0]

# Group duplicate technical replicates
pheno_grouped = (
    pheno_clean
    .groupby(["Base_ID", "Panel"])
    .agg({
        "PH": "mean",
        "EH": "mean"
    })
    .reset_index()
)

print("After grouping duplicates:", pheno_grouped.shape)


# ============================================
# 2️⃣ LOAD GENOTYPE (CORRECT WAY)
# ============================================

geno = pd.read_csv(
    "dataset/Geno_ASSO_NCRIPS_USP_28Ksnps.txt",
    sep=None,
    engine="python",
    index_col=0  # First column is sample ID
)

# Move ID from index to column
geno.reset_index(inplace=True)
geno.rename(columns={"index": "Full_name"}, inplace=True)

print("Genotype shape:", geno.shape)
print("First 5 genotype IDs:")
print(geno["Full_name"].head())


# ============================================
# 3️⃣ EXTRACT Base_ID FROM GENOTYPE
# ============================================

geno["Base_ID"] = geno["Full_name"].str.split(".").str[0]

print(geno[["Full_name", "Base_ID"]].head())


# ============================================
# 4️⃣ MERGE GENOTYPE + PHENOTYPE
# ============================================

merged = geno.merge(pheno_grouped, on="Base_ID")
print("Merged shape (with replicates):", merged.shape)


# ============================================
# 5️⃣ REMOVE GENOTYPE DUPLICATES (KEEP 1 PER Base_ID)
# ============================================

merged_unique = (
    merged
    .sort_values("Full_name")
    .groupby("Base_ID")
    .first()
    .reset_index()
)

print("Final unique merged shape:", merged_unique.shape)


# ============================================
# 6️⃣ SAVE FINAL DATASET
# ============================================

merged_unique.to_csv("final_merged_dataset.txt", sep="\t", index=False)

print("Final merged dataset saved as final_merged_dataset.txt")