from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
CORS(app)

# Configure PostgreSQL database
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:Pavan9567@localhost:5432/chatbot_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load GPT-2 model and tokenizer
MODEL_NAME = "gpt2"  # Open-source LLM
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Define models for suppliers and products
class Supplier(db.Model):
    __tablename__ = 'suppliers'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    contact_info = db.Column(db.Text, nullable=True)
    product_categories = db.Column(db.Text, nullable=True)

class Product(db.Model):
    __tablename__ = 'products'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    brand = db.Column(db.String(50), nullable=True)
    price = db.Column(db.Numeric, nullable=True)
    category = db.Column(db.String(50), nullable=True)
    description = db.Column(db.Text, nullable=True)
    supplier_id = db.Column(db.Integer, db.ForeignKey('suppliers.id'))

# Summarize supplier data using GPT-2
def summarize_with_gpt2(text):
    inputs = tokenizer(f"Summarize: {text}", return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('query', '').lower()

    # Handle product queries
    if "products under brand" in query:
        brand = query.split("brand")[-1].strip()
        products = Product.query.filter(Product.brand.ilike(f"%{brand}%")).all()
        if products:
            result = [{"name": p.name, "price": float(p.price), "category": p.category} for p in products]
            return jsonify(result)
        return jsonify({"error": "No products found for this brand."}), 404

    # Handle supplier queries
    elif "suppliers provide" in query:
        category = query.split("provide")[-1].strip()
        suppliers = Supplier.query.filter(Supplier.product_categories.ilike(f"%{category}%")).all()
        if suppliers:
            supplier_info = "\n".join(
                [f"Supplier: {s.name}, Contact: {s.contact_info}, Categories: {s.product_categories}" for s in suppliers]
            )
            summary = summarize_with_gpt2(supplier_info)
            return jsonify({"summary": summary})
        return jsonify({"error": "No suppliers found for this category."}), 404

    # Handle product details
    elif "details of product" in query:
        product_name = query.split("product")[-1].strip()
        product = Product.query.filter(Product.name.ilike(f"%{product_name}%")).first()
        if product:
            return jsonify({
                "name": product.name,
                "brand": product.brand,
                "price": float(product.price),
                "description": product.description
            })
        return jsonify({"error": "Product not found"}), 404

    return jsonify({"error": "Invalid query"}), 400

if __name__ == '__main__':
    app.run(debug=True)
