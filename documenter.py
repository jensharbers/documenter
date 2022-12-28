import openturns as ot
import pandas as pd
import numpy as np

def best_dist_docs_continuous(sample):
    sample = sample
    sample = sample.to_numpy()
    sample = sample[~np.isnan(sample)]
    sample = sample.reshape((-1, 1))

    tested_factories = ot.DistributionFactory.GetContinuousUniVariateFactories()
    best_model, best_bic = ot.FittingTest.BestModelBIC(sample, tested_factories)

    return str(best_model)


def best_dist_docs_discrete(sample):
    sample = sample
    sample = sample.to_numpy()
    sample = sample[~np.isnan(sample)]
    sample = sample.reshape((-1, 1))

    tested_factories = ot.DistributionFactory.GetDiscreteUniVariateFactories()
    best_model, best_bic = ot.FittingTest.BestModelBIC(sample, tested_factories)

    return str(best_model)



numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# Example section:
act = pd.read_csv("C:\\Users\\Jens Harbers\\Documents\\ACT_Collar00869_20200316171823.csv", sep=";",decimal=",", encoding="latin-1")
act["timestamp"] = pd.to_datetime(act['UTC_Date'] +" "+act['UTC_Time'], format="%d.%m.%Y %H:%M:%S")

df2 = act.select_dtypes(include=numerics)
df2.reset_index(drop=True)
# choose columns with at least two unique values
ids = df2.nunique().values>1
df2 = df2.loc[:,ids]

dist_list = []

for i in np.arange(len(df2.columns)):
    dist_list.append(best_dist_docs_continuous(df2.iloc[:,i]))

desc = df2.describe().T
desc["distributions"] = dist_list
desc["dtype"] = df2.dtypes
desc["kurtosis"]= df2.kurtosis()
desc["skew"]= df2.skew()
if len(act.select_dtypes(exclude=numerics)) > 0:
    df3 = act.select_dtypes(exclude=numerics)
    df3 = df3.apply(pd.Categorical)
    desc2 = df3.describe().T
    desc2["distributions"] = np.repeat(np.nan,len(act.select_dtypes(exclude=numerics).columns))
    desc2["dtype"] = df3.dtypes
    descs = pd.concat([desc,desc2])

# Dataset related information

def describer(subject, specific_subject_area, type_of_data, date_data_aquirement, 
              description_data_collection, data_source_location, data_accessability, 
              related_information, name_processor, name_owner, name_processing_file, 
              license, time_of_data, access_mode, keywords, ticket_number, version_tool_link, 
              version_number, title, authors, abstract, data_description, value_of_data, 
              experimental_design, material_and_methods, experimental_setup, experiments, 
              version, ethics_statement, declaration_of_competing_interest, 
              acknowledgments, credit_author_statement, supplementary_materials, 
              research_project, funding_source, references):
    """
    Describes various aspects of data, including subject, specific subject area, type of data, 
    date of data acquisition, data collection method, data source location, data accessibility, 
    and related information. Also includes information specific to the industry, such as the 
    processor and owner of the data, the processing file, license, and access mode. Additionally, 
    includes information relevant to scientific research, such as the title, authors, abstract, 
    data description, value of the data, experimental design, materials and methods, experimental 
    setup, number of experiments, version, ethics statement, declaration of competing interests, 
    acknowledgments, author contributions, supplementary materials, research project, funding 
    source, and references.
    
    Parameters
    ----------
    subject : str
        The subject of the data.
    specific_subject_area : str
        A more specific area within the subject.
    type_of_data : str
        The type of data, such as a file or raw data.
    date_data_aquirement : str
        The date the data was acquired.
    description_data_collection : str
        How the data was collected.
    data_source_location : str
        Where the data is stored.
    data_accessability : str
        The name and location of the repository where the data can be accessed.
    related_information : str
        Any additional information related to the data or research articles used.
    name_processor : str
        The name of the processor of the data, with an email address (applicable in the industry only).
    name_owner : str
        The name of the owner of the data, with an email address (applicable in the industry only).
    name_processing_file : str
        The name and origin of the processing file.
    license : str
        The license of the data, such as proprietary or a Creative Commons license.
    time_of_data : str
        The time at which the data was obtained or processed.
    access_mode : str
        The method of access, such as password, multi-factor authentication, or unprotected.
    keywords : str
        Related keywords to increase the chances of finding the data.
    ticket_number : str
        A JIRA ticket number or string (applicable in the industry only).
    version_tool_link : str
        A link to a version control tool such as GitHub or Bitbucket.
    version_number : str
        A Number of the experiment version
    title : str
        The title of the paper.
    authors : str
        The names of all authors.
    abstract : str
        The abstract of the article.
    data_description : str
        A description of the dataset.
    value_of_data : str
        The value of the dataset to science and to colleagues.
    experimental_design : str
        The experimental design used.
    material_and_methods : str
        A description of the materials and methods used.
    experimental_setup : str
        The setup of the experiment.
    experiments : str
        The number of experiments.
    version : str
        The version of the experiment.
    ethics_statement : str
        The ethics statement, required for animal experiments and in medicine.
    declaration_of_competing_interest : str
        The declaration of competing interests, such as funding issues.
    acknowledgments : str
        Individuals to thank.
    credit_author_statement : str
        The contributions of each author to the scientific process.
    supplementary_materials : str
        Any additional materials that can be linked.
    research_project : str
        The name of the research project (applicable in research only).
    funding_source : str
        The name of the sponsor (applicable in research only).
    references : str
        The references (more useful in research, optional otherwise).
    """
    # Specifications Table

     
data = {
    "Name": ["Subject", "Specific Subject Area", "Type of Data", "Date of Data Aquirement", 
             "Description of Data Collection", "Data Source Location", "Data Accessability", 
             "Related Information", "Name Processor", "Name Owner", "Name Processing File", 
             "License", "Time of Data", "Access Mode", "Keywords", "Ticket Number", 
             "Version Tool Link", "Version Number", "Title", "Authors", "Abstract", 
             "Data Description", "Value of Data", "Experimental Design", "Material and Methods", 
             "Experimental Setup", "Experiments", "Version", "Ethics Statement", 
             "Declaration of Competing Interest", "Acknowledgments", "CRediT Author Statement", 
             "Supplementary Materials", "Research Project", "Funding Source", "References"],
    "Value": [subject, specific_subject_area, type_of_data, date_data_aquirement, 
              description_data_collection, data_source_location, data_accessability, 
              related_information, name_processor, name_owner, name_processing_file, 
              license, time_of_data, access_mode, keywords, ticket_number, version_tool_link, 
              version_number, title, authors, abstract, data_description, value_of_data, 
              experimental_design, material_and_methods, experimental_setup, experiments, 
              version, ethics_statement, declaration_of_competing_interest, 
              acknowledgments, credit_author_statement,supplementary_materials,
              research_project,funding_source,references]
}

df = pd.DataFrame(data)
return df
 
# This code creates a dictionary with the keys "Name" and "Value", 
# and assigns the corresponding parameter values to the "Value" key. 
# The dictionary is then passed to the pd.DataFrame constructor to create a DataFrame. 
# The DataFrame is then returned by the function.
