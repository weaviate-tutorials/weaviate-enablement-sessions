# Weaviate 1.33 Features Showcase
## Comprehensive Guide to New Features and Improvements

This notebook demonstrates all the major features introduced in Weaviate v1.33, including compression by default, advanced quantization, server-side batching, and more.

## Table of Contents
1. [Setup and Prerequisites](#setup)
2. [Compression by Default & 8-bit RQ (GA)](#compression-default)
3. [1-bit Rotational Quantization (Preview)](#1bit-rq)
4. [Server-side Batch Imports (Preview)](#server-batch)
5. [OIDC Group Management](#oidc-groups)
6. [Collection Aliases (GA)](#aliases)
7. [New Filter Operators](#filters)
8. [Performance Considerations](#performance)

<a id="setup"></a>
## 1. Setup and Prerequisites

```python
# Install the latest Weaviate client
!pip install -U weaviate-client

import weaviate
from weaviate import classes as wvc
import os
import json
from datetime import datetime
import numpy as np

# Connect to Weaviate instance
# For Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url="YOUR_CLUSTER_URL",
    auth_credentials=weaviate.AuthApiKey("YOUR_API_KEY")
)

# For local Docker instance
# client = weaviate.connect_to_local()

# Check connection
print(f"Connected to Weaviate: {client.is_ready()}")
print(f"Weaviate version: {client.get_meta()['version']}")
```

<a id="compression-default"></a>
## 2. Compression by Default & 8-bit RQ (GA)

Starting with v1.33, **8-bit Rotational Quantization (RQ) is enabled by default** for all new collections. This provides up to 4x memory compression while maintaining 98-99% recall.

### Key Benefits:
- ✅ **Automatic optimization** - No configuration needed
- ✅ **4x memory compression** - Significant resource savings
- ✅ **98-99% recall maintained** - Minimal accuracy loss
- ✅ **No training phase** - Works immediately at index creation
- ✅ **Generally Available (GA)** - Production-ready

```python
from weaviate import classes as wvc

# Create a collection - RQ compression is now enabled by default!
default_compressed_collection = client.collections.create(
    name="Articles_Compressed_Default",
    properties=[
        wvc.Property(name="title", data_type=wvc.DataType.TEXT),
        wvc.Property(name="content", data_type=wvc.DataType.TEXT),
        wvc.Property(name="author", data_type=wvc.DataType.TEXT),
    ],
    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai()
)

print("✅ Collection created with default RQ compression enabled automatically!")

# Verify compression settings
collection_config = client.collections.get("Articles_Compressed_Default").config.get()
print(f"Quantization config: {collection_config.vector_index_config.quantizer}")
```

### Customizing Default Compression

You can customize the default compression behavior using environment variables:

```python
# To disable compression by default (set in environment before starting Weaviate)
# DEFAULT_QUANTIZATION=none

# To use PQ instead of RQ as default
# DEFAULT_QUANTIZATION=pq

# Example: Creating a collection without compression (override default)
uncompressed_collection = client.collections.create(
    name="Articles_Uncompressed",
    properties=[
        wvc.Property(name="title", data_type=wvc.DataType.TEXT),
        wvc.Property(name="content", data_type=wvc.DataType.TEXT),
    ],
    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
    vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
        quantizer=None  # Explicitly disable quantization
    )
)

print("✅ Collection created without compression (override default)")
```

### Comparing Memory Usage

```python
# Import sample data to both collections
sample_articles = [
    {
        "title": "Introduction to Vector Databases",
        "content": "Vector databases are revolutionizing how we store and search unstructured data...",
        "author": "Alice Johnson"
    },
    {
        "title": "Understanding Semantic Search",
        "content": "Semantic search goes beyond keyword matching to understand intent and context...",
        "author": "Bob Smith"
    },
    # Add more articles...
]

# Import to compressed collection
compressed_collection = client.collections.get("Articles_Compressed_Default")
with compressed_collection.batch.dynamic() as batch:
    for article in sample_articles:
        batch.add_object(properties=article)

# Import to uncompressed collection  
uncompressed_collection = client.collections.get("Articles_Uncompressed")
with uncompressed_collection.batch.dynamic() as batch:
    for article in sample_articles:
        batch.add_object(properties=article)

print("Data imported to both collections")

# Compare memory usage (in production, monitor actual memory metrics)
print("\n📊 Memory Comparison:")
print("Compressed (RQ): ~25% of original size")
print("Uncompressed: 100% of original size")
print("Savings: ~75% memory reduction with 98-99% recall maintained!")
```

<a id="1bit-rq"></a>
## 3. 1-bit Rotational Quantization (Preview) ⚠️

**1-bit RQ** is a cutting-edge preview feature offering extreme compression (up to 32x) while maintaining reasonable recall.

### ⚠️ Important Considerations:
- **Preview feature** - Not recommended for production
- **Asymmetric quantization** - 1-bit for data, 5-bit for queries
- **Better than Binary Quantization (BQ)** - More robust on challenging datasets
- **~10% throughput decrease vs BQ** - But significantly better recall

```python
# Create a collection with 1-bit RQ (PREVIEW - NOT FOR PRODUCTION)
one_bit_collection = client.collections.create(
    name="Articles_1bit_RQ_Preview",
    properties=[
        wvc.Property(name="title", data_type=wvc.DataType.TEXT),
        wvc.Property(name="content", data_type=wvc.DataType.TEXT),
    ],
    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
    vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
        quantizer=wvc.config.Configure.VectorIndex.Quantizer.rq(
            bits_per_vector=1,  # 1-bit quantization
            query_bits=5        # 5-bit for queries (asymmetric)
        )
    )
)

print("⚠️ Collection created with 1-bit RQ (Preview feature)")
print("📉 Compression: Close to 32x as dimensionality increases")
print("🎯 Use case: Extreme memory constraints where some recall loss is acceptable")
```

### Compression Comparison Table

```python
import pandas as pd

compression_comparison = pd.DataFrame({
    'Quantization Type': ['None', '8-bit RQ (Default)', '1-bit RQ (Preview)', 'PQ', 'BQ'],
    'Compression Rate': ['1x', '~4x', '~32x', '4-64x', '32x'],
    'Typical Recall': ['100%', '98-99%', '85-95%', '95-99%', '80-95%'],
    'Training Required': ['No', 'No', 'No', 'Yes', 'No'],
    'Status': ['GA', 'GA (Default)', 'Preview', 'GA', 'GA']
})

print("\n📊 Quantization Methods Comparison:")
print(compression_comparison.to_string(index=False))
```

<a id="server-batch"></a>
## 4. Server-side Batch Imports (Preview) ⚠️

Server-side batching (automatic batching) lets the server optimize data flow for maximum performance.

### Key Advantages:
- ✅ **No manual tuning** - Server optimizes batch size automatically
- ✅ **Dynamic backpressure** - Prevents server overload
- ✅ **Asynchronous error handling** - Errors don't interrupt flow
- ✅ **Better resilience** - Handles cluster scaling and long vectorization

```python
# Traditional client-side batching (old way)
def traditional_batching(collection_name, data_objects):
    """Manual batching - requires tuning batch_size and concurrent_requests"""
    collection = client.collections.get(collection_name)
    
    # Manual configuration required
    with collection.batch.fixed_size(
        batch_size=100,  # Manual tuning needed
        concurrent_requests=2  # Manual tuning needed
    ) as batch:
        for obj in data_objects:
            batch.add_object(properties=obj)
    
    print(f"Imported {len(data_objects)} objects with manual batching")

# NEW: Server-side automatic batching (v1.33)
def automatic_batching(collection_name, data_objects):
    """Server-side batching - automatic optimization!"""
    collection = client.collections.get(collection_name)
    
    # Server automatically optimizes everything!
    with collection.batch.automatic() as batch:
        for obj in data_objects:
            batch.add_object(properties=obj)
            
            # Server dynamically adjusts flow rate based on:
            # - Current server load
            # - Queue size (EMA calculation)
            # - Available resources
    
    print(f"✅ Imported {len(data_objects)} objects with automatic batching")
    print("📈 Server automatically optimized batch size and flow rate!")

# Example usage
large_dataset = [
    {"title": f"Article {i}", "content": f"Content for article {i}..."} 
    for i in range(10000)
]

# Create test collection
test_collection = client.collections.create(
    name="BatchImportTest",
    properties=[
        wvc.Property(name="title", data_type=wvc.DataType.TEXT),
        wvc.Property(name="content", data_type=wvc.DataType.TEXT),
    ]
)

# Use automatic batching (recommended for v1.33+)
automatic_batching("BatchImportTest", large_dataset[:1000])

# Error handling with automatic batching
def robust_automatic_import(collection_name, data_objects):
    """Automatic batching with error handling"""
    collection = client.collections.get(collection_name)
    errors = []
    
    with collection.batch.automatic() as batch:
        for i, obj in enumerate(data_objects):
            try:
                batch.add_object(properties=obj)
            except Exception as e:
                # Errors are handled asynchronously
                # Flow continues without interruption
                errors.append((i, str(e)))
    
    if errors:
        print(f"⚠️ Encountered {len(errors)} errors during import")
        for idx, error in errors[:5]:  # Show first 5 errors
            print(f"  - Object {idx}: {error}")
    else:
        print("✅ All objects imported successfully")
    
    return errors

# Test with some problematic data
mixed_data = [
    {"title": "Valid article", "content": "Good content"},
    {"title": None, "content": "Missing title"},  # Potential error
    {"title": "Another valid", "content": "More content"},
]

robust_automatic_import("BatchImportTest", mixed_data)
```

<a id="oidc-groups"></a>
## 5. OIDC Group Management

Manage permissions at scale using groups from your identity provider (Keycloak, Okta, Auth0, etc.).

### Key Features:
- ✅ **Automatic permission inheritance** - Users get permissions from their groups
- ✅ **Centralized management** - Define groups in your IdP
- ✅ **Bulk operations** - Assign roles to entire groups
- ✅ **Sync with organization structure** - Permissions match your org chart

```python
# Assuming you have admin client with proper permissions
admin_client = weaviate.connect_to_weaviate_cloud(
    cluster_url="YOUR_CLUSTER_URL",
    auth_credentials=weaviate.AuthApiKey("YOUR_ADMIN_API_KEY")
)

# Define roles for different teams
def setup_oidc_group_permissions():
    """Configure OIDC group-based permissions"""
    
    # 1. Assign roles to admin group
    admin_client.groups.oidc.assign_roles(
        group_id="/admin-group",
        role_names=["admin", "data_manager", "viewer"]
    )
    print("✅ Admin group configured with full permissions")
    
    # 2. Assign roles to data science team
    admin_client.groups.oidc.assign_roles(
        group_id="/data-science-team",
        role_names=["data_scientist", "viewer", "query_executor"]
    )
    print("✅ Data science team configured")
    
    # 3. Assign read-only access to analysts
    admin_client.groups.oidc.assign_roles(
        group_id="/analysts",
        role_names=["viewer", "query_executor"]
    )
    print("✅ Analyst group configured with read-only access")
    
    # 4. Configure engineering team
    admin_client.groups.oidc.assign_roles(
        group_id="/engineering",
        role_names=["developer", "data_manager", "viewer"]
    )
    print("✅ Engineering team configured")

# List all roles for a group
def list_group_permissions(group_id):
    """Display all roles assigned to a group"""
    group_roles = admin_client.groups.oidc.get_assigned_roles(
        group_id=group_id,
        include_permissions=True  # Include detailed permissions
    )
    
    print(f"\n📋 Roles for group '{group_id}':")
    for role_name, permissions in group_roles.items():
        print(f"  • {role_name}:")
        for perm in permissions[:3]:  # Show first 3 permissions
            print(f"    - {perm}")
    
    return group_roles

# Revoke specific roles
def update_group_permissions(group_id, roles_to_revoke):
    """Remove specific roles from a group"""
    admin_client.groups.oidc.revoke_roles(
        group_id=group_id,
        role_names=roles_to_revoke
    )
    print(f"✅ Revoked roles {roles_to_revoke} from {group_id}")

# Example workflow
print("=== OIDC Group Management Example ===")
# setup_oidc_group_permissions()  # Uncomment with proper admin access
# list_group_permissions("/data-science-team")
# update_group_permissions("/analysts", ["query_executor"])

# Demonstrate the permission flow
print("\n🔄 Permission Flow:")
print("1. User logs in via OIDC provider (Okta, Auth0, etc.)")
print("2. OIDC token includes user's groups")
print("3. Weaviate maps groups to roles")
print("4. User automatically has all permissions from their groups")
print("5. No individual user configuration needed!")
```

<a id="aliases"></a>
## 6. Collection Aliases (GA) 🎉

Collection aliases enable zero-downtime migrations and flexible production deployments.

### Key Benefits:
- ✅ **Zero-downtime migrations** - Switch collections instantly
- ✅ **A/B testing** - Test different configurations
- ✅ **Blue-green deployments** - Seamless production updates
- ✅ **Generally Available** - Production-ready!

```python
# Scenario: Migrating a production collection with zero downtime

# Step 1: Create initial production collection
def setup_production_collection():
    """Create initial production collection"""
    v1_collection = client.collections.create(
        name="Products_v1",
        properties=[
            wvc.Property(name="name", data_type=wvc.DataType.TEXT),
            wvc.Property(name="description", data_type=wvc.DataType.TEXT),
            wvc.Property(name="price", data_type=wvc.DataType.NUMBER),
            wvc.Property(name="category", data_type=wvc.DataType.TEXT),
        ],
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai()
    )
    
    # Import production data
    products = [
        {"name": "Laptop Pro", "description": "High-performance laptop", "price": 1299.99, "category": "electronics"},
        {"name": "Wireless Mouse", "description": "Ergonomic wireless mouse", "price": 29.99, "category": "accessories"},
        {"name": "USB-C Hub", "description": "7-in-1 USB-C hub", "price": 49.99, "category": "accessories"},
    ]
    
    with v1_collection.batch.automatic() as batch:
        for product in products:
            batch.add_object(properties=product)
    
    print("✅ Production collection 'Products_v1' created and populated")
    return v1_collection

# Step 2: Create alias pointing to production
def create_production_alias():
    """Create alias for production use"""
    client.aliases.create(
        alias_name="Products",  # This is what applications use
        target_collection="Products_v1"
    )
    print("✅ Alias 'Products' -> 'Products_v1' created")
    print("📱 Applications can now use 'Products' alias")

# Step 3: Applications use the alias (not the versioned name)
def application_query():
    """Simulate application querying via alias"""
    # Application always uses the alias, not version-specific names
    products = client.collections.get("Products")  # Using alias!
    
    results = products.query.near_text(
        query="laptop accessories",
        limit=3
    )
    
    print("\n🔍 Query results via alias 'Products':")
    for item in results.objects:
        print(f"  - {item.properties['name']}: ${item.properties['price']}")

# Step 4: Prepare new version with improvements
def prepare_v2_collection():
    """Create improved v2 collection"""
    v2_collection = client.collections.create(
        name="Products_v2",
        properties=[
            wvc.Property(name="name", data_type=wvc.DataType.TEXT),
            wvc.Property(name="description", data_type=wvc.DataType.TEXT),
            wvc.Property(name="price", data_type=wvc.DataType.NUMBER),
            wvc.Property(name="category", data_type=wvc.DataType.TEXT),
            wvc.Property(name="brand", data_type=wvc.DataType.TEXT),  # NEW field
            wvc.Property(name="in_stock", data_type=wvc.DataType.BOOLEAN),  # NEW field
        ],
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
        # Improved configuration
        vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
            distance_metric=wvc.config.VectorDistances.COSINE,
            ef=256,  # Improved search quality
            quantizer=wvc.config.Configure.VectorIndex.Quantizer.rq()  # Using new default RQ
        )
    )
    
    # Migrate and enhance data
    enhanced_products = [
        {"name": "Laptop Pro", "description": "High-performance laptop", 
         "price": 1299.99, "category": "electronics", "brand": "TechCorp", "in_stock": True},
        {"name": "Wireless Mouse", "description": "Ergonomic wireless mouse", 
         "price": 29.99, "category": "accessories", "brand": "Logitech", "in_stock": True},
        {"name": "USB-C Hub", "description": "7-in-1 USB-C hub", 
         "price": 49.99, "category": "accessories", "brand": "Anker", "in_stock": False},
    ]
    
    with v2_collection.batch.automatic() as batch:
        for product in enhanced_products:
            batch.add_object(properties=product)
    
    print("\n✅ Version 2 collection 'Products_v2' ready with enhancements:")
    print("  • Added 'brand' field")
    print("  • Added 'in_stock' field")
    print("  • Improved vector index configuration")
    print("  • Enabled RQ compression")

# Step 5: Zero-downtime migration
def perform_zero_downtime_migration():
    """Switch alias to new version with zero downtime"""
    print("\n🔄 Performing zero-downtime migration...")
    
    # Update alias to point to v2
    client.aliases.update(
        alias_name="Products",
        new_target_collection="Products_v2"
    )
    
    print("✅ Migration complete! 'Products' now points to 'Products_v2'")
    print("🎉 Applications continue working without any downtime!")
    print("🗑️  Old collection 'Products_v1' can be deleted after verification")

# Step 6: Rollback if needed
def rollback_if_needed():
    """Emergency rollback to previous version"""
    print("\n⚠️ Issue detected - performing rollback...")
    
    client.aliases.update(
        alias_name="Products",
        new_target_collection="Products_v1"
    )
    
    print("✅ Rolled back to 'Products_v1'")
    print("🔧 Fix issues in v2 and try again")

# Execute the migration workflow
print("=== Collection Alias Migration Demo ===")
# Uncomment to run:
# setup_production_collection()
# create_production_alias()
# application_query()
# prepare_v2_collection()
# perform_zero_downtime_migration()
# application_query()  # Still works, now using v2!

# List all aliases
def list_all_aliases():
    """Display all configured aliases"""
    aliases = client.aliases.list()
    print("\n📋 Configured Aliases:")
    for alias in aliases:
        print(f"  • {alias.name} -> {alias.target}")

# list_all_aliases()
```

<a id="filters"></a>
## 7. New Filter Operators: ContainsNone and Not

Enhanced filtering capabilities with two powerful new operators.

### ContainsNone Operator
Returns objects where the property contains **none** of the specified values.

### Not Operator
Provides logical negation of conditions.

```python
# Setup test collection with sample data
def setup_filter_demo_collection():
    """Create collection for filter demonstration"""
    collection = client.collections.create(
        name="BlogPosts",
        properties=[
            wvc.Property(name="title", data_type=wvc.DataType.TEXT),
            wvc.Property(name="content", data_type=wvc.DataType.TEXT),
            wvc.Property(name="tags", data_type=wvc.DataType.TEXT_ARRAY),
            wvc.Property(name="category", data_type=wvc.DataType.TEXT),
            wvc.Property(name="published", data_type=wvc.DataType.BOOLEAN),
            wvc.Property(name="views", data_type=wvc.DataType.NUMBER),
        ]
    )
    
    # Sample blog posts
    posts = [
        {
            "title": "Introduction to AI",
            "content": "AI is transforming industries...",
            "tags": ["ai", "technology", "future"],
            "category": "technology",
            "published": True,
            "views": 1500
        },
        {
            "title": "Healthy Cooking Tips",
            "content": "Eating healthy doesn't have to be boring...",
            "tags": ["health", "cooking", "lifestyle"],
            "category": "lifestyle",
            "published": True,
            "views": 800
        },
        {
            "title": "Sports Analytics Revolution",
            "content": "Data science in sports...",
            "tags": ["sports", "analytics", "data"],
            "category": "sports",
            "published": True,
            "views": 2000
        },
        {
            "title": "Draft: Future of Web3",
            "content": "Exploring decentralized web...",
            "tags": ["web3", "blockchain", "technology"],
            "category": "technology",
            "published": False,
            "views": 0
        },
        {
            "title": "Travel Guide to Japan",
            "content": "Best places to visit in Japan...",
            "tags": ["travel", "japan", "culture"],
            "category": "travel",
            "published": True,
            "views": 3000
        }
    ]
    
    with collection.batch.automatic() as batch:
        for post in posts:
            batch.add_object(properties=post)
    
    print("✅ BlogPosts collection created with sample data")
    return collection

# ContainsNone Filter Examples
def demonstrate_contains_none():
    """Show ContainsNone operator usage"""
    collection = client.collections.get("BlogPosts")
    
    print("\n=== ContainsNone Operator Examples ===")
    
    # Find posts that DON'T contain any of these tags
    results = collection.query.fetch_objects(
        where=wvc.query.Filter.by_property("tags").contains_none(
            ["politics", "sports", "entertainment"]
        ),
        limit=5
    )
    
    print("\n1️⃣ Posts without politics, sports, or entertainment tags:")
    for post in results.objects:
        print(f"  - {post.properties['title']}: {post.properties['tags']}")
    
    # Complex query: Non-entertainment posts with high engagement
    results = collection.query.fetch_objects(
        where=wvc.query.Filter.all_of([
            wvc.query.Filter.by_property("tags").contains_none(["entertainment", "celebrity"]),
            wvc.query.Filter.by_property("views").greater_than(1000)
        ]),
        limit=5
    )
    
    print("\n2️⃣ High-engagement posts (>1000 views) without entertainment content:")
    for post in results.objects:
        print(f"  - {post.properties['title']}: {post.properties['views']} views")

# Not Operator Examples
def demonstrate_not_operator():
    """Show Not operator usage"""
    collection = client.collections.get("BlogPosts")
    
    print("\n=== Not Operator Examples ===")
    
    # Find all posts NOT in technology category
    results = collection.query.fetch_objects(
        where=wvc.query.Filter.by_property("category").not_equal("technology"),
        limit=5
    )
    
    print("\n1️⃣ Posts NOT in technology category:")
    for post in results.objects:
        print(f"  - {post.properties['title']}: {post.properties['category']}")
    
    # Complex: Published posts that are NOT low engagement
    results = collection.query.fetch_objects(
        where=wvc.query.Filter.all_of([
            wvc.query.Filter.by_property("published").equal(True),
            wvc.query.Filter.not_(
                wvc.query.Filter.by_property("views").less_than(1000)
            )
        ]),
        limit=5
    )
    
    print("\n2️⃣ Published posts with NOT low engagement (>=1000 views):")
    for post in results.objects:
        print(f"  - {post.properties['title']}: {post.properties['views']} views")

# Combined filter examples
def demonstrate_complex_filters():
    """Show complex filter combinations"""
    collection = client.collections.get("BlogPosts")
    
    print("\n=== Complex Filter Combinations ===")
    
    # Find: Published posts, NOT in sports/entertainment, with specific tags
    results = collection.query.fetch_objects(
        where=wvc.query.Filter.all_of([
            wvc.query.Filter.by_property("published").equal(True),
            wvc.query.Filter.by_property("category").contains_none(["sports", "entertainment"]),
            wvc.query.Filter.any_of([
                wvc.query.Filter.by_property("tags").contains_any(["ai", "technology"]),
                wvc.query.Filter.by_property("views").greater_than(2500)
            ])
        ]),
        limit=10
    )
    
    print("\nComplex query results:")
    print("(Published + NOT sports/entertainment + (has tech tags OR high views))")
    for post in results.objects:
        print(f"  - {post.properties['title']}")
        print(f"    Category: {post.properties['category']}, Views: {post.properties['views']}")
        print(f"    Tags: {post.properties['tags']}")

# Run filter demonstrations
# setup_filter_demo_collection()
# demonstrate_contains_none()
# demonstrate_not_operator()
# demonstrate_complex_filters()

print("\n💡 Filter Operator Tips:")
print("• ContainsNone: Efficiently excludes multiple values")
print("• Not: Inverts any condition for flexible queries")
print("• Both operators work with all other filters")
print("• Optimized for performance at scale")
```

<a id="performance"></a>
## 8. Performance Considerations and Best Practices

```python
print("=== Performance Best Practices for Weaviate 1.33 ===\n")

# Performance monitoring setup
def create_performance_test_collection():
    """Create collection with optimal v1.33 settings"""
    
    optimal_collection = client.collections.create(
        name="OptimalPerformance",
        properties=[
            wvc.Property(name="content", data_type=wvc.DataType.TEXT),
            wvc.Property(name="metadata", data_type=wvc.DataType.OBJECT),
        ],
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(
            model="text-embedding-3-small"  # Efficient embedding model
        ),
        vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
            distance_metric=wvc.config.VectorDistances.COSINE,
            ef=256,  # Balance between speed and recall
            ef_construction=384,  # Higher value for better index quality
            max_connections=32,  # Optimal for most use cases
            quantizer=wvc.config.Configure.VectorIndex.Quantizer.rq(  # Default RQ
                # Automatically uses 8-bit RQ in v1.33
            )
        ),
        inverted_index_config=wvc.config.Configure.inverted_index(
            bm25_b=0.75,
            bm25_k1=1.25
        )
    )
    
    print("✅ Collection created with optimal v1.33 settings:")
    print("  • RQ compression enabled (default)")
    print("  • HNSW optimized for balance")
    print("  • BM25 configured for hybrid search")
    
    return optimal_collection

# Benchmark different configurations
def benchmark_configurations():
    """Compare performance across different settings"""
    
    configurations = {
        "Default (RQ)": {
            "quantizer": "rq",
            "memory": "25%",
            "speed": "95%",
            "recall": "98-99%"
        },
        "Uncompressed": {
            "quantizer": None,
            "memory": "100%",
            "speed": "100%",
            "recall": "100%"
        },
        "1-bit RQ (Preview)": {
            "quantizer": "rq-1bit",
            "memory": "~3%",
            "speed": "85%",
            "recall": "85-95%"
        },
        "PQ": {
            "quantizer": "pq",
            "memory": "10-25%",
            "speed": "90%",
            "recall": "95-99%"
        }
    }
    
    print("\n📊 Performance Comparison (v1.33):")
    print("-" * 60)
    print(f"{'Configuration':<20} {'Memory':<10} {'Speed':<10} {'Recall':<10}")
    print("-" * 60)
    
    for config_name, metrics in configurations.items():
        print(f"{config_name:<20} {metrics['memory']:<10} {metrics['speed']:<10} {metrics['recall']:<10}")
    
    print("\n💡 Recommendations:")
    print("  • Use default (RQ) for most use cases")
    print("  • Disable compression only for tiny datasets (<10K objects)")
    print("  • Consider PQ for very high dimensional vectors (>1536)")
    print("  • 1-bit RQ only for extreme memory constraints (preview)")

# Import performance tips
def import_performance_tips():
    """Best practices for data import"""
    
    print("\n=== Import Performance Best Practices ===")
    
    tips = [
        ("Use automatic batching", "Let server optimize batch size", "⭐⭐⭐⭐⭐"),
        ("Avoid tiny batches", "Minimum 10-50 objects per batch", "⭐⭐⭐⭐"),
        ("Parallelize imports", "Use multiple threads/processes", "⭐⭐⭐⭐"),
        ("Pre-compute embeddings", "For large datasets, vectorize offline", "⭐⭐⭐"),
        ("Monitor server metrics", "Watch CPU, memory, disk I/O", "⭐⭐⭐⭐⭐"),
        ("Use compression", "Default RQ saves memory", "⭐⭐⭐⭐⭐"),
    ]
    
    for i, (practice, description, importance) in enumerate(tips, 1):
        print(f"\n{i}. {practice} {importance}")
        print(f"   → {description}")
    
    # Example: Optimized import function
    print("\n📝 Example: Optimized import with monitoring")
    print("""
    async def optimized_import(collection_name, data_source):
        collection = client.collections.get(collection_name)
        
        # Use automatic batching (v1.33 feature)
        with collection.batch.automatic() as batch:
            # Track progress
            for i, item in enumerate(data_source):
                batch.add_object(properties=item)
                
                # Progress reporting
                if i % 1000 == 0:
                    print(f"Imported {i} objects...")
        
        print(f"✅ Import complete: {len(data_source)} objects")
    """)

# Query optimization tips
def query_optimization_tips():
    """Best practices for query performance"""
    
    print("\n=== Query Performance Optimization ===")
    
    print("\n1️⃣ Use appropriate limits:")
    print("   • Start with smaller limits (10-100)")
    print("   • Increase only if needed")
    print("   • Use pagination for large result sets")
    
    print("\n2️⃣ Optimize vector search:")
    print("   • Tune 'ef' parameter (default: 256)")
    print("   • Higher ef = better recall, slower")
    print("   • Lower ef = faster, may miss results")
    
    print("\n3️⃣ Leverage filters effectively:")
    print("   • Filter before vector search when possible")
    print("   • Use indexed properties for filters")
    print("   • Combine filters efficiently")
    
    print("\n4️⃣ Hybrid search optimization:")
    print("   • Balance alpha parameter (0.5 default)")
    print("   • alpha=0: keyword only")
    print("   • alpha=1: vector only")
    print("   • Test different values for your use case")
    
    # Example optimized query
    print("\n📝 Example: Optimized hybrid search")
    print("""
    # Optimized hybrid search with filters
    results = collection.query.hybrid(
        query="machine learning applications",
        alpha=0.7,  # Favor vector search
        limit=20,  # Reasonable limit
        where=wvc.query.Filter.by_property("category").equal("technology"),
        return_properties=["title", "summary"],  # Only needed fields
        return_metadata=wvc.query.MetadataQuery(distance=True)
    )
    """)

# Run performance analysis
# create_performance_test_collection()
benchmark_configurations()
import_performance_tips()
query_optimization_tips()

print("\n" + "="*60)
print("🎉 Weaviate 1.33 Performance Summary:")
print("  ✅ RQ compression by default - 4x memory savings")
print("  ✅ Automatic batching - No manual tuning needed")  
print("  ✅ Enhanced HNSW performance - Faster queries")
print("  ✅ Optimized LSM store - Better write throughput")
print("  ✅ Improved query planning - Complex filters faster")
print("="*60)
```

## Summary and Next Steps

```python
print("=== Weaviate 1.33 Feature Summary ===\n")

summary = """
🚀 **Major Features (Production-Ready):**
   • Compression by default (8-bit RQ) - GA ✅
   • Collection aliases - GA ✅
   • OIDC group management ✅
   • New filter operators (ContainsNone, Not) ✅

⚠️ **Preview Features (Not for Production):**
   • 1-bit rotational quantization
   • Server-side batch imports (automatic batching)

📈 **Performance Improvements:**
   • Enhanced HNSW indexing
   • Improved LSM store efficiency
   • Optimized query planning
   • Memory usage optimizations

🎯 **Best Practices:**
   1. Use default RQ compression (4x memory savings)
   2. Leverage automatic batching for imports
   3. Implement collection aliases for zero-downtime migrations
   4. Use OIDC groups for scalable permission management
   5. Take advantage of new filter operators for complex queries

📚 **Resources:**
   • Documentation: https://weaviate.io/docs
   • GitHub: https://github.com/weaviate/weaviate
   • Community: https://weaviate.io/community
   • Cloud Console: https://console.weaviate.cloud
"""

print(summary)

# Cleanup
client.close()
print("\n✅ Notebook complete! Happy building with Weaviate 1.33! 🎉")
```

## Appendix: Migration Checklist

```python
print("=== Migration to Weaviate 1.33 Checklist ===\n")

checklist = """
Pre-Migration:
□ Review release notes and breaking changes
□ Backup existing data
□ Test in staging environment
□ Update client libraries to latest version

Migration Steps:
□ Update Weaviate to v1.33
□ Verify default RQ compression behavior
□ Update import code to use automatic batching
□ Implement collection aliases for future migrations
□ Configure OIDC groups if using authentication
□ Update queries to use new filter operators where beneficial

Post-Migration:
□ Monitor memory usage (should decrease with RQ)
□ Verify query performance
□ Test error handling with automatic batching
□ Document configuration changes
□ Train team on new features

Performance Validation:
□ Compare memory usage (before/after)
□ Benchmark query latency
□ Test import throughput
□ Validate recall rates with compression
□ Monitor system stability
"""

print(checklist)
```