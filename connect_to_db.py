import psycopg2


class TheiaConnect(object):
    def __init__(self, dbname, user, host, port, password):
        self.conn = self.connect_db(dbname, user, host, port, password)
        self.cur = self.conn.cursor()

    '''Connect to the database.'''

    def connect_db(self, dbname, user, host, port, password):
        try:
            self.conn = psycopg2.connect(dbname=dbname, user=user,
                                         host=host,
                                         port=port, password=password)
        except Exception:
            print("I am unable to connect to the database")

        print("--connected---")
        return self.conn

    '''Make a change to the database.'''

    def make_change(self, changes):
        try:
            self.cur.execute(changes)
            self.conn.commit()
            print("Executed Task")
        except Exception:
            print("Unable to execute SQL command")

    '''Create a table in the database.'''

    def create_table(self, schema, tablename, datatypes):
        try:
            self.cur.execute("""create table if not exists %s.%s (%s)""" %
                             (schema, tablename, datatypes))
            self.conn.commit()
            print("Table created!")
        except Exception:
            print("Unable to create table")

# connector.make_change("Create table sample1 (name varchar(255), age int)")
# connector.make_change("Insert into sample1 (name, age) \
# values ('DAVIDKES', -9), ('Kobe', 21)")
