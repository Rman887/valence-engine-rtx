def can_build(env, platform):
    # The subdivision builder (D122) makes refined meshes for 3D characters at import.
    return not env["disable_3d"]


def configure(env):
    pass


def get_doc_classes():
    return [
        "SubdivisionBuilder",
    ]


def get_doc_path():
    return "doc_classes"
