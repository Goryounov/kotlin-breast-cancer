package smile

import org.apache.commons.csv.CSVFormat
import smile.data.formula.Formula
import smile.io.Read
import smile.validation.CrossValidation
import smile.classification.GradientTreeBoost

fun main() {
    // Загрузка датасета Breast Cancer
    var dsFileFormat = CSVFormat.DEFAULT.builder()
        .setHeader()
        .setSkipHeaderRecord(true)
        .setDelimiter(',')
        .build()
    val dataset = Read.csv(
        "/Users/nikita/development/university/kotlin/kotlin-for-data-science/frameworks/frameworks/src/main/kotlin/smile/Cancer_Data.csv",
        dsFileFormat
    )

    // Преобразуем целевую переменную (diagnosis) в 1 и 0
    for (i in 0 until dataset.nrow()) {
        val diagnosis = dataset.getString(i, 1)
        val classId = if (diagnosis == "M") "1" else "0"
        dataset.set(i, 1, classId)
    }
    println(dataset)

    val f = Formula.lhs("diagnosis")

    // Обучаем GradientTreeBoost
    val res = CrossValidation.classification(
        10, f, dataset,
        { formula, data -> GradientTreeBoost.fit(formula, data) })
    println(res)
}