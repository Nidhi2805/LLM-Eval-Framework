from db.models import init_db, get_session, Project, Document

init_db()
print("Database initialized (eval_platform.db created).")

# Smoke test: create a project with one document, read it back
session = get_session()

# Clean slate for the test
session.query(Project).filter_by(name="smoke-test").delete()
session.commit()

proj = Project(name="smoke-test", description="verifying the schema works")
proj.documents.append(Document(external_id="doc_001", title="Test Doc",
                               text="This is a test document."))
session.add(proj)
session.commit()

# Read it back
loaded = session.query(Project).filter_by(name="smoke-test").first()
print(f"Created project '{loaded.name}' (id={loaded.id}) "
      f"with {len(loaded.documents)} document(s):")
for d in loaded.documents:
    print(f"   {d.external_id}: {d.title}")

session.close()